训练流程：  

>     1. 训练基于swinir的单目预训练模型 model_swinir  
         ```
         默认 2块GPU 进行训练
         python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main_train_alone.py --opt ./opt/sr_flickr1024_swinir.json  --dist True
         ```

>     2. 转换 model_swinir 模型为 swinipassr 的预训练模型 pre_swinipassr 
         ```
         挑选 swinir 结果中 psnr 值 超过 23.70+ 的模型， 进行模型转换
         python swinir_to_swinipassr.py --swinir_path ./superresolution/sr_flickr1024_swinir/models/*_G.pth --swinipassr_path ./pretrained/pre_swinipassr.pth
         ```
      
>     3. 训练基于swinipassr的双目训练模型 model_swinipassr 
         ```
         python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main_train_double.py --opt ./opt/ssr_flickr1024_swinipassr.json  --dist True
         ```

>     4. 取 swinipassr 的最后5个模型进行平均，生成最终的finetune模型
         ```
         python swinipassr_to_plus.py
         ```

>     5. 对 model_swinipassr 进行 finetune， 获取模型 model_swinipassr_plus
         ```
         python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main_train_double.py --opt ./opt/ssr_flickr1024_swinipassr_plus.json  --dist True
         ```

>     6. 在观察到 ssr_flickr1024_swinipassr_plus 训练过程中出现过拟合的情况时，手动停止模型，并对过拟前模型进行平均，得到最终模型。 例如该模型训练日志如下所示：
         ```
         input:
               cat ./superresolution/sr_flickr1024_swinir_plus/train.log | grep "Validation"

         output:
               22-03-25 10:06:11.457 : [Validation] iter:5000, Average PSNR : 23.96dB
               22-03-25 11:30:21.689 : [Validation] iter:10000, Average PSNR : 23.95dB
               22-03-25 12:54:19.929 : [Validation] iter:15000, Average PSNR : 23.96dB
               22-03-25 14:18:30.008 : [Validation] iter:20000, Average PSNR : 23.97dB
               22-03-25 15:42:32.057 : [Validation] iter:25000, Average PSNR : 23.97dB
               22-03-25 17:06:29.785 : [Validation] iter:30000, Average PSNR : 23.96dB
               22-03-25 18:30:27.290 : [Validation] iter:35000, Average PSNR : 23.97dB
               22-03-25 19:54:29.104 : [Validation] iter:40000, Average PSNR : 23.96dB
               22-03-25 21:18:28.774 : [Validation] iter:45000, Average PSNR : 23.97dB
               22-03-25 22:42:33.977 : [Validation] iter:50000, Average PSNR : 23.95dB
               22-03-26 00:06:36.079 : [Validation] iter:55000, Average PSNR : 23.91dB
         ```
         则应该停止模型，并选择 iter50000 之前的所有模型进行平均
     
>     7. 根据 步骤5 得到的iter数，来生成最终模型. 模型位于./pretrained/swinipassr_final.pth
         ```
         python swinipassr_final.py --iter 50000
         ```

比赛结果复现流程：

>     1. mkdir ./final_models & mkdir ./pretrained & mkdir ./superresolution & mkdir ./logits
>     2. rm -rf ./pretrained/* & rm -rf ./superresolution/*  (必须执行)
>     3. 修改 ./opt/sr_flickr1024_swinir.json, ./opt/ssr_flickr1024_swinipassr.json, ./opt/ssr_flickr1024_swinipassr_plus.json
         ```
         ...
         , "H_size": 96
         ...
         , "img_size": 24
         , "window_size": 12
         , "img_range": 1.0 
         , "depths": [9, 9, 9, 9, 9, 9] 
         , "embed_dim": 180 
         , "num_heads": [9, 9, 9, 9, 9, 9] 
         ```
>     4. 执行训练流程
>     5. mv ./pretrained/swinipassr_final.pth ./final_models/P24W12D9E180H9.pth
>     6. python main_test_double.py --task classical_sr --scale 4 --training_patch_size 24 --model_path ./final_models/P24W12D9E180H9.pth --folder_lq test_path
>     7. rm -rf ./pretrained/* & rm -rf ./superresolution/*  （必须执行）
>     8. 修改 ./opt/sr_flickr1024_swinir.json, ./opt/ssr_flickr1024_swinipassr.json, ./opt/ssr_flickr1024_swinipassr_plus.json
         ```
         ...
         , "H_size": 96
         ...
         , "img_size": 24
         , "window_size": 8
         , "img_range": 1.0 
         , "depths": [9, 9, 9, 9, 9, 9] 
         , "embed_dim": 180 
         , "num_heads": [9, 9, 9, 9, 9, 9] 
         ```
>     9. 执行训练流程
>     10. mv /pretrained/swinipassr_final.pth ./final_models/P24W8D9E180H9.pth

      ...

>     python logits_to_images.py --folder_lq test_path

>     zip ./results.zip ./results/swinir_classical_sr_x4/*.png