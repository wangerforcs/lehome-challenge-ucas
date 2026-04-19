# 🤖 1st  Challenge Submission: Dynamic MoE Router with Specialized Experts

**Team Name:** 
**Affiliation:** UCAS
**Track:** Simulation Challenge 

---

## 1. 📑 Method Overview
Our solution implements a **Dynamic Mixture-of-Experts (MoE)** framework to address the multi-category garment manipulation task in the  challenge. 

- **Vision-based Router:** We utilize a fine-tuned ResNet18 classifier to determine the specific garment category (`pant_long`, `pant_short`, `top_long`, `top_short`) from the initial RGB observation frame.
- **Specialized Experts:** Four independent **ACT/PI05/XVLA** policies are deployed as "experts." Each expert is trained and optimized exclusively on one garment category to ensure high-precision manipulation and robust handling of specific cloth geometries.
- **Dynamic Policy Switching:** At the start of each evaluation episode, the system identifies the target type and dynamically routes the control flow to the corresponding expert, ensuring the most suitable policy is active for the task at hand.

## 2. 🏠 Repository Structure
```text
.
├── screenshots/                                        # Evaluation result screenshots
├── scripts/eval_policy/example_participant_policy.py  # Core MoE Logic & Dynamic Routing
├── outputs/classifier/garment_classifier_resnet18.pth # Fine-tuned ResNet18 Weights
├── outputs/train/                                     # Specialized Expert Weights
│   ├── pant_long_best
│   ├── pant_short_best
│   ├── top_long_best                                  
│   └── top_short_best
└── Datasets/example/                                  # Metadata for Normalization
    ├── pant_long_merged/meta/stats.json
    ├── pant_short_merged/meta/stats.json
    ├── top_long_merged/meta/stats.json
    └── top_short_merged/meta/stats.json
```

## 3. 🧮 Evaluation Instructions
To evaluate our policy, please ensure the official Isaac Sim and LeRobot environment is properly configured. Our policy is registered under the type `custom`.

### 3.1 Retrieve Policy and Weights
Please clone this Hugging Face repository into your evaluation workspace:
```bash
git lfs install
git clone https://huggingface.co/bigbangoslab/policies
```

Please ensure that all files referenced under policies/ also exist at the corresponding paths under the original project root, or are mapped there via symbolic links.

### 3.2 Recommended Execution (By Category)

**👖 For Pants (裤装类):**
```bash
# Evaluate Long Pants
python -m scripts.eval --policy_type custom --garment_type pant_long --enable_cameras --device cpu

# Evaluate Short Pants
python -m scripts.eval --policy_type custom --garment_type pant_short --enable_cameras --device cpu
```

**👕 For Tops (上装类):**
```bash
# Evaluate Long Sleeve Tops
python -m scripts.eval --policy_type custom --garment_type top_long --enable_cameras --device cpu

# Evaluate Short Sleeve Tops
python -m scripts.eval --policy_type custom --garment_type top_short --enable_cameras --device cpu
```

## 4. 📧 Contact Information
- **Primary Contact:** Zhanbo Wang (beachbum9527@gmail.com)


