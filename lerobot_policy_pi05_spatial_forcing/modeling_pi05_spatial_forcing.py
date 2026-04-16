from __future__ import annotations

import builtins
import logging
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    PI05Pytorch,
    compute_layer_complete,
    make_att_2d_masks,
)

from .configuration_pi05_spatial_forcing import PI05SpatialForcingConfig

from vggt.heads.utils import custom_pooling
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import preprocess_images_from_openpi

T = TypeVar("T", bound="PI05SpatialForcingPolicy")


class AlignProjector(nn.Module):
    """Project VLA image tokens to the VGGT feature dimension and compute cosine alignment loss."""

    def __init__(self, llm_dim: int, vggt_dim: int, use_vlm_norm: bool = False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(llm_dim, 2 * vggt_dim, bias=True)
        self.fc2 = nn.Linear(2 * vggt_dim, 2 * vggt_dim, bias=True)
        self.act_fn1 = nn.GELU()
        self.vlm_norm = nn.LayerNorm(llm_dim) if use_vlm_norm else None
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def align_dimension(self, llm_embedding: torch.Tensor) -> torch.Tensor:
        target_dtype = self.fc1.weight.dtype
        target_device = self.fc1.weight.device
        llm_embedding = llm_embedding.to(device=target_device, dtype=target_dtype)
        if self.vlm_norm is not None:
            self.vlm_norm = self.vlm_norm.to(device=target_device, dtype=target_dtype)
            llm_embedding = self.vlm_norm(llm_embedding)
        projected_features = self.fc1(llm_embedding)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features

    def forward(self, llm_emb: torch.Tensor, target_emb: torch.Tensor, align_mask: torch.Tensor) -> torch.Tensor:
        llm_emb = self.align_dimension(llm_emb)
        target_emb = target_emb.to(device=llm_emb.device, dtype=llm_emb.dtype)
        align_mask = align_mask.to(device=llm_emb.device)
        llm_emb = F.normalize(llm_emb, dim=-1)
        target_emb = F.normalize(target_emb, dim=-1)
        cosine = (llm_emb * target_emb).sum(dim=-1)
        valid = cosine[align_mask]
        if valid.numel() == 0:
            return torch.zeros((), dtype=llm_emb.dtype, device=llm_emb.device)
        return 1 - valid.mean()


class PI05SpatialForcingPytorch(PI05Pytorch):
    """PI05 with an auxiliary spatial alignment loss against VGGT features."""

    def __init__(self, config: PI05SpatialForcingConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        self.config: PI05SpatialForcingConfig = config
        self.spatial_forcing_image_keys = tuple(config.spatial_forcing_image_keys or ())
        self._align_debug_logged = False

        llm_dim = self.paligemma_with_expert.paligemma.config.vision_config.projection_dim
        self.align_projector = AlignProjector(
            llm_dim=llm_dim,
            vggt_dim=config.vggt_dim,
            use_vlm_norm=config.use_vlm_norm,
        )
        self.vggt_model = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
            feature_only=True,
        )
        self._load_vggt_weights()
        self._freeze_vggt()

    def _load_vggt_weights(self) -> None:
        if not self.config.vggt_weight_path:
            if self.config.align_loss_coeff > 0:
                raise ValueError(
                    "vggt_weight_path is required when align_loss_coeff > 0 for pi05_spatial_forcing"
                )
            return
        vggt_path = Path(self.config.vggt_weight_path) / "model.pt"
        if not vggt_path.exists():
            raise FileNotFoundError(f"VGGT weight file not found at {vggt_path}")
        state = torch.load(vggt_path, map_location="cpu")
        self.vggt_model.load_state_dict(state, strict=False)
        logging.info("Loaded VGGT weights from %s", vggt_path)

    def _freeze_vggt(self) -> None:
        self.vggt_model.eval()
        for param in self.vggt_model.parameters():
            param.requires_grad = False

    def _log_align_debug_once(
        self,
        images_for_align: list[Tensor],
        img_emb_token_counts: list[int],
        vggt_input: Tensor,
        vggt_hidden: Tensor,
        pooled_vggt_hidden: Tensor,
        vision_hidden: Tensor,
        patch_start_idx: int,
    ) -> None:
        if self._align_debug_logged:
            return

        vggt_bs, vggt_views, vggt_patch_tokens, vggt_dim = vggt_hidden.shape
        pooled_tokens = pooled_vggt_hidden.shape[1]
        vla_tokens = vision_hidden.shape[1]
        image_shapes = [tuple(img.shape) for img in images_for_align]
        vggt_input_shape = tuple(vggt_input.shape)
        patch_size = int(self.vggt_model.patch_size)
        input_height, input_width = vggt_input.shape[-2:]
        patch_h = input_height // patch_size
        patch_w = input_width // patch_size
        pooled_tokens_per_view = pooled_tokens // max(vggt_views, 1)

        logging.info(
            "[Spatial Forcing Debug] image_keys=%s, image_shapes=%s, vla_img_tokens_per_view=%s, "
            "vla_total_img_tokens=%d, vggt_input_shape=%s, vggt_patch_tokens_per_view=%d, "
            "vggt_pooled_tokens_per_view=%d, pooled_total_tokens=%d, patch_start_idx=%d, "
            "vggt_patch_grid=(%d,%d), vggt_dim=%d",
            self.spatial_forcing_image_keys if self.spatial_forcing_image_keys else "<all>",
            image_shapes,
            img_emb_token_counts,
            vla_tokens,
            vggt_input_shape,
            vggt_patch_tokens,
            pooled_tokens_per_view,
            pooled_tokens,
            patch_start_idx,
            patch_h,
            patch_w,
            vggt_dim,
        )
        self._align_debug_logged = True

    def _validate_align_shapes(
        self,
        images_for_align: list[Tensor],
        img_emb_token_counts: list[int],
        vggt_hidden: Tensor,
        pooled_vggt_hidden: Tensor,
        vision_hidden: Tensor,
    ) -> None:
        num_views = len(images_for_align)
        if num_views == 0:
            raise ValueError("Spatial forcing requires at least one image view for alignment")

        if len(img_emb_token_counts) != num_views:
            raise ValueError(
                f"Image token count bookkeeping mismatch: got {len(img_emb_token_counts)} counts for {num_views} views"
            )

        if len(set(img_emb_token_counts)) != 1:
            raise ValueError(
                "PI05 image token counts are inconsistent across views: "
                f"{img_emb_token_counts}. Current alignment assumes the same token count per aligned view."
            )

        if vggt_hidden.shape[1] != num_views:
            raise ValueError(
                f"VGGT view count mismatch: got {vggt_hidden.shape[1]} views from VGGT for {num_views} aligned images"
            )

        vla_total_tokens = vision_hidden.shape[1]
        pooled_total_tokens = pooled_vggt_hidden.shape[1]
        if pooled_total_tokens != vla_total_tokens:
            raise ValueError(
                "Token count mismatch between PI05 vision hidden states and pooled VGGT features: "
                f"PI05 has {vla_total_tokens} tokens, VGGT has {pooled_total_tokens} tokens after pooling"
            )

        if vla_total_tokens % num_views != 0:
            raise ValueError(
                f"PI05 vision token count {vla_total_tokens} is not divisible by number of aligned views {num_views}"
            )

        if pooled_total_tokens % num_views != 0:
            raise ValueError(
                f"Pooled VGGT token count {pooled_total_tokens} is not divisible by number of aligned views {num_views}"
            )

        per_view_vla_tokens = vla_total_tokens // num_views
        if per_view_vla_tokens != img_emb_token_counts[0]:
            raise ValueError(
                "Per-view PI05 token count mismatch: "
                f"vision_hidden implies {per_view_vla_tokens}, but embed_image produced {img_emb_token_counts[0]}"
            )

    def _embed_prefix(self, images, img_masks, tokens, masks):
        embs = []
        pad_masks = []
        att_masks = []
        total_image_token_count = 0
        image_token_counts = []

        for img, img_mask in zip(images, img_masks, strict=True):
            def image_embed_func(img_tensor):
                return self.paligemma_with_expert.embed_image(img_tensor)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]
            total_image_token_count += num_img_embs
            image_token_counts.append(num_img_embs)
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        def lang_embed_func(tokens_tensor):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens_tensor)
            return lang_emb * (lang_emb.shape[-1] ** 0.5)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))
        return embs, pad_masks, att_masks, total_image_token_count, image_token_counts

    def _resolve_vla_align_layer(self, num_layers: int) -> int:
        layer = int(self.config.vla_layers_align)
        if layer < 0:
            layer += num_layers
        if layer < 0 or layer >= num_layers:
            raise ValueError(
                f"vla_layers_align={self.config.vla_layers_align} resolves to {layer}, "
                f"but valid range is [0, {num_layers - 1}]"
            )
        return layer

    def _forward_suffix_and_collect_prefix_hidden(
        self,
        prefix_embs: Tensor,
        suffix_embs: Tensor,
        att_2d_masks_4d: Tensor,
        position_ids: Tensor,
        adarms_cond: Tensor,
    ) -> tuple[Tensor, Tensor]:
        models = [
            self.paligemma_with_expert.paligemma.language_model,
            self.paligemma_with_expert.gemma_expert.model,
        ]
        num_layers = self.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers
        target_layer_idx = self._resolve_vla_align_layer(num_layers)

        inputs_embeds = [prefix_embs, suffix_embs]
        use_gradient_checkpointing = (
            hasattr(self.paligemma_with_expert.gemma_expert.model, "gradient_checkpointing")
            and self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing
            and self.training
        ) or (
            hasattr(self.paligemma_with_expert, "gradient_checkpointing")
            and self.paligemma_with_expert.gradient_checkpointing
            and self.training
        ) or (self.gradient_checkpointing_enabled and self.training)

        prefix_hidden = None
        for layer_idx in range(num_layers):
            if use_gradient_checkpointing:
                inputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_layer_complete,
                    layer_idx,
                    inputs_embeds,
                    att_2d_masks_4d,
                    position_ids,
                    [None, adarms_cond],
                    use_reentrant=False,
                    preserve_rng_state=False,
                    paligemma=self.paligemma_with_expert.paligemma,
                    gemma_expert=self.paligemma_with_expert.gemma_expert,
                )
            else:
                inputs_embeds = compute_layer_complete(
                    layer_idx,
                    inputs_embeds,
                    att_2d_masks_4d,
                    position_ids,
                    [None, adarms_cond],
                    paligemma=self.paligemma_with_expert.paligemma,
                    gemma_expert=self.paligemma_with_expert.gemma_expert,
                )
            if layer_idx == target_layer_idx:
                prefix_hidden = inputs_embeds[0]

        if prefix_hidden is None:
            raise RuntimeError("Failed to capture VLA prefix hidden state for alignment")

        def compute_final_norms(hidden_inputs, cond):
            outputs_embeds = []
            for i, hidden_states in enumerate(hidden_inputs):
                out_emb, _ = models[i].norm(hidden_states, cond=cond[i])
                outputs_embeds.append(out_emb)
            return outputs_embeds

        if use_gradient_checkpointing:
            outputs_embeds = torch.utils.checkpoint.checkpoint(
                compute_final_norms,
                inputs_embeds,
                [None, adarms_cond],
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            outputs_embeds = compute_final_norms(inputs_embeds, [None, adarms_cond])

        suffix_out = outputs_embeds[1]
        return suffix_out, prefix_hidden

    def _compute_align_loss(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        vision_hidden: Tensor,
        image_token_counts: list[int],
    ) -> Tensor:
        if self.config.align_loss_coeff <= 0:
            return torch.zeros((), dtype=torch.float32, device=images[0].device)

        images_for_align = images
        masks_for_align = img_masks
        image_token_counts_for_align = image_token_counts
        if self.spatial_forcing_image_keys:
            max_count = min(len(self.spatial_forcing_image_keys), len(images))
            images_for_align = images[:max_count]
            masks_for_align = img_masks[:max_count]
            image_token_counts_for_align = image_token_counts[:max_count]

        vggt_input = preprocess_images_from_openpi([(img + 1.0) / 2.0 for img in images_for_align], mode="crop")

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if images[0].device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx, torch.no_grad():
            vggt_output = self.vggt_model(vggt_input)

        vggt_hidden = vggt_output["features"][-1]
        patch_start_idx = vggt_output["patch_start_idx"]
        original_img = vggt_input
        vggt_hidden = vggt_hidden[:, :, patch_start_idx:, :]

        height, width = original_img.shape[-2:]
        patch_h, patch_w = height // self.vggt_model.patch_size, width // self.vggt_model.patch_size
        pooled_vggt_hidden = custom_pooling(
            vggt_hidden,
            (patch_h, patch_w),
            (height, width),
            vision_hidden,
            self.config.pooling_func,
            self.config.use_vggt_pe,
        )
        pooled_vggt_hidden = pooled_vggt_hidden.to(dtype=torch.float32)
        vision_hidden = vision_hidden.to(dtype=torch.float32)
        self._validate_align_shapes(
            images_for_align,
            image_token_counts_for_align,
            vggt_hidden,
            pooled_vggt_hidden,
            vision_hidden,
        )
        self._log_align_debug_once(
            images_for_align,
            image_token_counts_for_align,
            vggt_input,
            vggt_hidden,
            pooled_vggt_hidden,
            vision_hidden,
            patch_start_idx,
        )

        tokens_per_img = vision_hidden.shape[1] // len(images_for_align)
        img_masks_stack = torch.stack(masks_for_align, dim=1)
        align_mask = torch.repeat_interleave(img_masks_stack, repeats=tokens_per_img, dim=1)
        return self.align_projector(vision_hidden, pooled_vggt_hidden, align_mask)

    def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None):
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, total_image_token_count, image_token_counts = self._embed_prefix(
            images, img_masks, tokens, masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        suffix_out, prefix_hidden = self._forward_suffix_and_collect_prefix_hidden(
            prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self._apply_checkpoint(self.action_out_proj, suffix_out)
        action_losses = F.mse_loss(u_t, v_t, reduction="none")
        vision_hidden = prefix_hidden[:, :total_image_token_count, :]
        align_loss = self._compute_align_loss(images, img_masks, vision_hidden, image_token_counts)
        return action_losses, align_loss


class PI05SpatialForcingPolicy(PI05Policy):
    config_class = PI05SpatialForcingConfig
    name = "pi05_spatial_forcing"

    def __init__(self, config: PI05SpatialForcingConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = PI05SpatialForcingPytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        return super().from_pretrained(
            pretrained_name_or_path=pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return super().predict_action_chunk(batch, **kwargs)

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)

        action_losses, align_loss = self.model.forward(images, img_masks, tokens, masks, actions)
        original_action_dim = self.config.output_features[ACTION].shape[0]
        action_losses = action_losses[:, :, :original_action_dim]
        action_loss = action_losses.mean()
        weighted_align_loss = self.config.align_loss_coeff * align_loss
        total_loss = action_loss + weighted_align_loss

        loss_dict = {
            "loss_per_dim": action_losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
            "action_loss": action_loss.item(),
            "align_loss": align_loss.item(),
            "weighted_align_loss": weighted_align_loss.item(),
            "loss": total_loss.item(),
        }

        if reduction == "none":
            per_sample_action_loss = action_losses.mean(dim=(1, 2))
            per_sample_total_loss = per_sample_action_loss + self.config.align_loss_coeff * align_loss
            return per_sample_total_loss, loss_dict

        return total_loss, loss_dict
