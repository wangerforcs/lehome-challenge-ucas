pant_short 244/250 这个效果在我旧版测试最好
pant_long 164/250 这个测试就很垃圾
top_short 214/250 这个测试也很垃圾
top_long 206/250 测试还行
一共828条数据

第二次，数量差不多831，但是tmd筛去的不一样
pant_short 245/250
pant_long 158/250
top_short 217/250
top_long 211/250

filter2/2的
pant_short: 232 / 250
pant_long: 116 / 250
top_short: 166 / 250
top_long: 170 / 250



pi05二次训练后的结果

chunksize=10，n_action_steps=10
|task|seen|unseen|all|
|:--:|:--:|:--:|:--:|
|top_long|62%|70%|63.3%|
|top_short|20%|0%|16.7%|
|pant_long|14%|0%|11.7%|
|pant_short|94%|50%|86.6%|


chunksize=10，n_action_steps=5
|task|seen|unseen|all|
|:--:|:--:|:--:|:--:|
|top_long|62%|70%|63.3%|
|top_short|20%|0%|16.7%|
|pant_long|0.0|0.0||
|pant_short|94%|50%|86.6%|


filter repley1的

|task|seen|unseen|all|
|:--:|:--:|:--:|:--:|
|top_long|76%|80%|76.7%|
|top_short|50%|10%|43.3%|
|pant_long|38%|0%|31.67%|
|pant_short|96%|40%|86.7%|

这个的checkpoint就是/home/wzb/challenges/lehome-challenge/outputs/train/pi05_filterd/checkpoints/010000/pretrained_model

这里其实可以看出了，明显整理过的数据集效果更好(不过我训练的batch之类的都变了，能说明什么吗，我也不知道)


我觉得pant long效果很差的原因是，tmd他把pant long当成short来做了，首先空抓，因为short短，其次不折叠。
然后我觉得确实可能是区分度不够高，导致老是识别错，然后要不这两个分开搞吧，或者单独训练pi试试，我想先试试pant long和top short，要不先top short吧


top short不抓袖子(不折叠)到底是怎么回事，即使单独训练也是这样，我感觉可能是因为action chunk?




我的路径是pi05topshort.yaml 6000步
然后pi05topshortcontinue.yaml 6000步
chunk size50
30->10->5, 5, 5这种顺序
Top_Short_Seen_5: Success Rate = 80.00%, Avg Return = 143.74
第一个折叠半身差一点

chunk size50 action10
Top_Short_Seen_5: Success Rate = 60.00%, Avg Return = 146.83
第一个袖子折的不好，最后一个折叠半身不好


chunk size50 action5
Top_Short_Seen_5: Success Rate = 80.00%, Avg Return = 145.46
第一个本来袖子好了，折中间袖子又烂回去了

貌似这个按照50训了之后，seen没啥直接折叠的情况了，都会去折袖子


上面的初步感觉就是action5更好，于是我用5测了全部的(006000的)

seen最后两个拉了，其实并不是unseen对应的，而是一个难看的衣服，皱巴巴的，反而类似于女露脐装的是中间几个的效果还行

新的全量训练的

9000 steps
seen8 
Top_Short_Seen_8: Success Rate = 0.00%, Avg Return = 160.00

seen9
Average Return: 167.76 ± 69.49
Success Rate: 20.00%

30/10/5/5/5
Top_Short_Seen_8: Success Rate = 0.00%, Avg Return = 126.96

15000 steps chunk5
Top_Short_Seen_8: Success Rate = 60.00%, Avg Return = 119.52
14很烂

Top_Short_Seen_9: Success Rate = 40.00%, Avg Return = 158.22
测了两次
145很烂
124很烂





top short是filtered chunk10
continue是接top short chunk50
continuefull接top short chunk50但是全量数据


现在top short最好的是home/wzb/challenges/lehome-challenge/outputs/train/top_short_pi05_continue_full/checkpoints/015000/pretrained_model 

top long最好的是
/home/wzb/challenges/lehome-challenge/outputs/train/pi05_filterd/checkpoints/010000/pretrained_model

pant short用/home/wzb/challenges/lehome-challenge/outputs/train/pi05_ultra2/checkpoints/024000/pretrained_model