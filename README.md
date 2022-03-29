零、(1) 还没有上传模型文件  
   (2) 是否需要zip压缩命令
   (3) 还差P得模型

一、测试流程 & 测试结果复现:
>   (1) 创建训练测试所需文件夹
>   ```
>   mkdir ./logits
>   mkdir ./pretrained
>   mkdir ./results
>   mkdir ./superresolution
>   mkdir ./final_models
>   ```
>   (2) 从 URL1/URL2 下载所有模型至 ./final_models
>   (3) 执行如下命令以推理  
>   ```
>   python main_test_double.py --task classical_sr --scale 4 --training_patch_size 24 --window_size 8 --model_path ./final_models/P24W8D9E180H9.pth --folder_lq "Test Dataset Path" 
>   python main_test_double.py --task classical_sr --scale 4 --training_patch_size 24 --window_size 12 --model_path ./final_models/P24W12D9E180H9.pth --folder_lq "Test Dataset Path"  
>   python main_test_double.py --task classical_sr --scale 4 --training_patch_size 24 --window_size 12 --model_path ./final_models/P24W12D9E180H9P.pth --folder_lq "Test Dataset Path"  
>   ```
>   (4) 获取最终模型结果
>   ```
>   python logits_to_images.py
>   ```
>   (5) ./results 文件夹内，即为最终提交结果

二、训练流程 & 结果复现  
>   (1) 清除测试历史数据
>   ```
>   rm -rf ./logits/*
>   rm -rf ./pretrained/*
>   rm -rf ./results/*
>   rm -rf ./superresolution/*
>   rm -rf ./final_models/*
>   ```
>   (2) P24W8D9E180H9.pth 模型训练过程：  
>    >  a. 清除历史文件信息
>    >  ```
>    >  rm -rf ./pretrained/* 
>    >  rm -rf ./superresolution/* 
>    >  ```
>    >  b. 单目 Swinir 模型配置修改, 打开编辑配置文件 "./opt/sr_flickr1024_swinir.json", 并修改其对应内容为：
>    >  ```
>    >  ...
>    >  
>    >  , "H_size": 96
>    >  
>    >  ...
>    >  
>    >  , "img_size": 24
>    >  , "window_size": 8
>    >  , "img_range": 1.0 
>    >  , "depths": [9, 9, 9, 9, 9, 9] 
>    >  , "embed_dim": 180 
>    >  , "num_heads": [9, 9, 9, 9, 9, 9] 
>    >  
>    >  ...
>    >  
>    >  ```
>    >  c. 训练单目模型 Swinir
>    >  ```
>    >  # 默认使用2块GPU进行训练
>    >  python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main_train_alone.py --opt ./opt/sr_flickr1024_swinir.json  --dist True
>    >  ```
>    >  d. 转换 Swinir 模型为双目 SwiniPassr 的预训练模型
>    >  ```
>    >  python swinir_to_swinipassr.py
>    >  ``` 
>    >  e. 可以在 "./pretrained" 中查看，是否存在 "pre_swinipassr.pth" 文件，如果存在则继续执行，否则重复步骤d  
>    >  f. 双目 SwiniPassr 模型配置修改, 打开编辑配置文件 "./opt/sr_flickr1024_swinipassr.json", 并修改其对应内容为：
>    >  ```
>    >  ...
>    >  
>    >  , "H_size": 96
>    >  
>    >  ...
>    >  
>    >  , "img_size": 24
>    >  , "window_size": 8
>    >  , "img_range": 1.0 
>    >  , "depths": [9, 9, 9, 9, 9, 9] 
>    >  , "embed_dim": 180 
>    >  , "num_heads": [9, 9, 9, 9, 9, 9] 
>    >  
>    >  ...
>    >  
>    >  , "pretrained": "./pretrained/pre_swinipassr.pth"
>    >  , "param_keys": []
>    >  
>    >  , "weight_sr": 1.0
>    >  , "weight_photo": 0.1
>    >  , "weight_smooth": 0.01
>    >  , "weight_cycle": 0.01
>    >  , "weight_cons": 0.01
>    >  ```
>    >  g. 训练双目模型 SwiniPassr
>    >  ```
>    >  # 默认使用2块GPU进行训练
>    >  python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main_train_double.py --opt ./opt/ssr_flickr1024_swinipassr.json  --dist True
>    >  ```
>    >  h. 转换训练得到 SwiniPassr 模型为 SwiniPassr++ 的预训练模型
>    >  ```
>    >  python swinipassr_to_plus.py
>    >  ``` 
>    >  i. 可以在 "./pretrained" 中查看，是否存在 "pre_swinipassr_plus.pth" 文件，如果存在则继续执行，否则重复步骤h  
>    >  j. 双目 SwiniPassr++ 模型配置修改, 打开编辑配置文件 "./opt/sr_flickr1024_swinipassr_plus.json", 并修改其对应内容为：
>    >  ```
>    >  ...
>    >  
>    >  , "H_size": 96
>    >  
>    >  ...
>    >  
>    >  , "img_size": 24
>    >  , "window_size": 8
>    >  , "img_range": 1.0 
>    >  , "depths": [9, 9, 9, 9, 9, 9] 
>    >  , "embed_dim": 180 
>    >  , "num_heads": [9, 9, 9, 9, 9, 9] 
>    >  
>    >  ...
>    >  
>    >  , "pretrained": "./pretrained/pre_swinipassr_plus.pth"
>    >  , "param_keys": []
>    >  
>    >  , "weight_sr": 1.0
>    >  , "weight_photo": 0.1
>    >  , "weight_smooth": 0.01
>    >  , "weight_cycle": 0.1
>    >  , "weight_cons": 0.1
>    >  ```
>    >  k. 训练双目模型 SwiniPassr++
>    >  ```
>    >  # 默认使用2块GPU进行训练
>    >  python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main_train_double.py --opt ./opt/ssr_flickr1024_swinipassr_plus.json  --dist True
>    >  ```
>    >  l. SwiniPassr++ 模型会快速拟合，所以取过拟合前模型进行融合。根据训练日志决定融合的模型，例如训练日志如下所示，则选取 iter 50000之前的模型进行融合 (iter是5000的整数倍)  
>    >  ```
>    >  input:
>    >        cat ./superresolution/ssr_flickr1024_swinipassr_plus/train.log | grep "Validation"
>    >
>    >  output:
>    >        22-03-25 10:06:11.457 : [Validation] iter:5000, Average PSNR : 23.96dB
>    >        22-03-25 11:30:21.689 : [Validation] iter:10000, Average PSNR : 23.95dB
>    >        22-03-25 12:54:19.929 : [Validation] iter:15000, Average PSNR : 23.96dB
>    >        22-03-25 14:18:30.008 : [Validation] iter:20000, Average PSNR : 23.97dB
>    >        22-03-25 15:42:32.057 : [Validation] iter:25000, Average PSNR : 23.97dB
>    >        22-03-25 17:06:29.785 : [Validation] iter:30000, Average PSNR : 23.96dB
>    >        22-03-25 18:30:27.290 : [Validation] iter:35000, Average PSNR : 23.97dB
>    >        22-03-25 19:54:29.104 : [Validation] iter:40000, Average PSNR : 23.96dB
>    >        22-03-25 21:18:28.774 : [Validation] iter:45000, Average PSNR : 23.97dB
>    >        22-03-25 22:42:33.977 : [Validation] iter:50000, Average PSNR : 23.95dB
>    >        22-03-26 00:06:36.079 : [Validation] iter:55000, Average PSNR : 23.91dB
>    >        22-03-26 00:06:36.079 : [Validation] iter:55000, Average PSNR : 23.89dB
>    >  ```
>    >  m. 根据步骤k确定的iter, 进行 SwiniPassr++ 模型融合
>    >  ```
>    >  python swinipassr_final.py --path ./superresolution/ssr_flickr1024_swinipassr_plus --iter "步骤k确定的iter"
>    >  ```
>    >  n. 可以在 "./pretrained" 中查看，是否存在 "swinipassr_final.pth" 文件，如果存在则继续执行，否则重复步骤l
>    >  o. 移动并重命名 SwiniPassr++ 模型
>    >  ```
>    >  mv ./pretrained/swinipassr_final.pth ./final_models/P24W8D9E180H9.pth
>    >  ```
>    >
>   (3) P24W12D9E180H9.pth 模型训练过程：  
>    >  a. 清除历史文件信息
>    >  ```
>    >  rm -rf ./pretrained/* 
>    >  rm -rf ./superresolution/* 
>    >  ```
>    >  b. 单目 Swinir 模型配置修改, 打开编辑配置文件 "./opt/sr_flickr1024_swinir.json", 并修改其对应内容为：
>    >  ```
>    >  ...
>    >  
>    >  , "H_size": 96
>    >  
>    >  ...
>    >  
>    >  , "img_size": 24
>    >  , "window_size": 12
>    >  , "img_range": 1.0 
>    >  , "depths": [9, 9, 9, 9, 9, 9] 
>    >  , "embed_dim": 180 
>    >  , "num_heads": [9, 9, 9, 9, 9, 9] 
>    >  
>    >  ...
>    >  
>    >  ```
>    >  c. 训练单目模型 Swinir
>    >  ```
>    >  # 默认使用2块GPU进行训练
>    >  python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main_train_alone.py --opt ./opt/sr_flickr1024_swinir.json  --dist True
>    >  ```
>    >  d. 转换 Swinir 模型为双目 SwiniPassr 的预训练模型
>    >  ```
>    >  python swinir_to_swinipassr.py
>    >  ``` 
>    >  e. 可以在 "./pretrained" 中查看，是否存在 "pre_swinipassr.pth" 文件，如果存在则继续执行，否则重复步骤d  
>    >  f. 双目 SwiniPassr 模型配置修改, 打开编辑配置文件 "./opt/sr_flickr1024_swinipassr.json", 并修改其对应内容为：  
>    >  ```
>    >  ...
>    >  
>    >  , "H_size": 96
>    >  
>    >  ...
>    >  
>    >  , "img_size": 24
>    >  , "window_size": 12
>    >  , "img_range": 1.0 
>    >  , "depths": [9, 9, 9, 9, 9, 9] 
>    >  , "embed_dim": 180 
>    >  , "num_heads": [9, 9, 9, 9, 9, 9] 
>    >  
>    >  ...
>    >  
>    >  , "pretrained": "./pretrained/pre_swinipassr.pth"
>    >  , "param_keys": []
>    >  
>    >  , "weight_sr": 1.0
>    >  , "weight_photo": 0.1
>    >  , "weight_smooth": 0.01
>    >  , "weight_cycle": 0.01
>    >  , "weight_cons": 0.01
>    >  ```
>    >  g. 训练双目模型 SwiniPassr
>    >  ```
>    >  # 默认使用2块GPU进行训练
>    >  python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main_train_double.py --opt ./opt/ssr_flickr1024_swinipassr.json  --dist True
>    >  ```
>    >  h. 转换训练得到 SwiniPassr 模型为 SwiniPassr++ 的预训练模型
>    >  ```
>    >  python swinipassr_to_plus.py
>    >  ``` 
>    >  i. 可以在 "./pretrained" 中查看，是否存在 "pre_swinipassr_plus.pth" 文件，如果存在则继续执行，否则重复步骤h  
>    >  j. 双目 SwiniPassr++ 模型配置修改, 打开编辑配置文件 "./opt/sr_flickr1024_swinipassr_plus.json", 并修改其对应内容为：
>    >  ```
>    >  ...
>    >  
>    >  , "H_size": 96
>    >  
>    >  ...
>    >  
>    >  , "img_size": 24
>    >  , "window_size": 12
>    >  , "img_range": 1.0 
>    >  , "depths": [9, 9, 9, 9, 9, 9] 
>    >  , "embed_dim": 180 
>    >  , "num_heads": [9, 9, 9, 9, 9, 9] 
>    >  
>    >  ...
>    >  
>    >  , "pretrained": "./pretrained/pre_swinipassr_plus.pth"
>    >  , "param_keys": []
>    >  
>    >  , "weight_sr": 1.0
>    >  , "weight_photo": 0.1
>    >  , "weight_smooth": 0.01
>    >  , "weight_cycle": 0.1
>    >  , "weight_cons": 0.1
>    >  ```
>    >  k. 训练双目模型 SwiniPassr++
>    >  ```
>    >  # 默认使用2块GPU进行训练
>    >  python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main_train_double.py --opt ./opt/ssr_flickr1024_swinipassr_plus.json  --dist True
>    >  ```
>    >  l. SwiniPassr++ 模型会快速拟合，所以取过拟合前模型进行融合。根据训练日志决定融合的模型，例如训练日志如下所示，则选取 iter 50000之前的模型进行融合
>    >  ```
>    >  input:
>    >        cat ./superresolution/ssr_flickr1024_swinipassr_plus/train.log | grep "Validation"
>    >
>    >  output:
>    >        22-03-25 10:06:11.457 : [Validation] iter:5000, Average PSNR : 23.96dB
>    >        22-03-25 11:30:21.689 : [Validation] iter:10000, Average PSNR : 23.95dB
>    >        22-03-25 12:54:19.929 : [Validation] iter:15000, Average PSNR : 23.96dB
>    >        22-03-25 14:18:30.008 : [Validation] iter:20000, Average PSNR : 23.97dB
>    >        22-03-25 15:42:32.057 : [Validation] iter:25000, Average PSNR : 23.97dB
>    >        22-03-25 17:06:29.785 : [Validation] iter:30000, Average PSNR : 23.96dB
>    >        22-03-25 18:30:27.290 : [Validation] iter:35000, Average PSNR : 23.97dB
>    >        22-03-25 19:54:29.104 : [Validation] iter:40000, Average PSNR : 23.96dB
>    >        22-03-25 21:18:28.774 : [Validation] iter:45000, Average PSNR : 23.97dB
>    >        22-03-25 22:42:33.977 : [Validation] iter:50000, Average PSNR : 23.95dB
>    >        22-03-26 00:06:36.079 : [Validation] iter:55000, Average PSNR : 23.91dB
>    >        22-03-26 00:06:36.079 : [Validation] iter:55000, Average PSNR : 23.89dB
>    >  ```
>    >  m. 根据步骤k确定的iter, 进行 SwiniPassr++ 模型融合
>    >  ```
>    >  python swinipassr_final.py --path ./superresolution/ssr_flickr1024_swinipassr_plus --iter "步骤k确定的iter"
>    >  ```
>    >  n. 可以在 "./pretrained" 中查看，是否存在 "swinipassr_final.pth" 文件，如果存在则继续执行，否则重复步骤l
>    >  o. 移动并重命名 SwiniPassr++ 模型
>    >  ```
>    >  mv ./pretrained/swinipassr_final.pth ./final_models/P24W12D9E180H9.pth
>    >  ```
>    >
>   (4) P24W12D9E180H9P.pth 模型训练过程：  
>    > a. [⏰][⏰][⏰] 训练 P24W12D9E180H9P 模型，依赖于 P24W12D9E180H9 历史，所以不需要清除步骤  
>    > b. 将训练好的双目 SwiniPassr 模型转换为 双目双分辨率 SwiniPapssr 模型
>    > ```
>    > python swinipassr_to_swinipapssr.py
>    > ```
>    > c. 可以在 "./pretrained" 中查看，是否存在 "pre_swinipapssr.pth" 文件，如果存在则继续执行，否则重复步骤b  
>    > d. 双目双分辨率 SwiniPapssr 模型配置修改, 打开编辑配置文件 "./opt/sr_flickr1024_swinipapssr.json", 并修改其对应内容为：
>    >  ```
>    >  ...
>    >  
>    >  , "H_size": 96
>    >  
>    >  ...
>    >  
>    >  , "img_size": 24
>    >  , "window_size": 12
>    >  , "img_range": 1.0 
>    >  , "depths": [9, 9, 9, 9, 9, 9] 
>    >  , "embed_dim": 180 
>    >  , "num_heads": [9, 9, 9, 9, 9, 9] 
>    >  
>    >  ...
>    >  
>    >  , "pretrained": "./pretrained/pre_swinipapssr.pth"
>    >  , "param_keys": []
>    >  
>    >  , "weight_sr": 1.0
>    >  , "weight_photo": 0.1
>    >  , "weight_smooth": 0.01
>    >  , "weight_cycle": 0.1
>    >  , "weight_cons": 0.1
>    >  ```
>    >  e. 训练双目双分辨率模型 SwiniPapssr
>    >  ```
>    >  # 默认使用2块GPU进行训练
>    >  python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main_train_double.py --opt ./opt/ssr_flickr1024_swinipapssr.json  --dist True
>    >  ```
>    >  f. SwiniPapssr 仅需少量训练就能起到提升效果，训练iter仅需超过50000次即可
>    >  g. 双目双分辨率 SwiniPapssr++ 模型配置修改, 打开编辑配置文件 "./opt/sr_flickr1024_swinipapssr_plus.json", 并修改其对应内容为：
>    >  ```
>    >  ...
>    >  
>    >  , "H_size": 96
>    >  
>    >  ...
>    >  
>    >  , "img_size": 24
>    >  , "window_size": 12
>    >  , "img_range": 1.0 
>    >  , "depths": [9, 9, 9, 9, 9, 9] 
>    >  , "embed_dim": 180 
>    >  , "num_heads": [9, 9, 9, 9, 9, 9] 
>    >  
>    >  ...
>    >  
>    >  , "pretrained": "./superresolution/ssr_flickr1024_swinipapssr/models/50000_E.pth"
>    >  , "param_keys": []
>    >  
>    >  , "weight_sr": 1.0
>    >  , "weight_photo": 0.1
>    >  , "weight_smooth": 0.01
>    >  , "weight_cycle": 0.1
>    >  , "weight_cons": 0.1
>    >  ```
>    >  h. 训练双目双分辨率模型 SwiniPapssr++
>    >  ```
>    >  # 默认使用2块GPU进行训练
>    >  python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 main_train_double.py --opt ./opt/ssr_flickr1024_swinipapssr_plus.json  --dist True
>    >  ```
>    >  i. SwiniPapssr++ 模型会快速拟合，所以取过拟合前模型进行融合。根据训练日志决定融合的模型，例如训练日志如下所示，则选取 iter 50000之前的模型进行融合
>    >  ```
>    >  input:
>    >        cat ./superresolution/ssr_flickr1024_swinipapssr_plus/train.log | grep "Validation"
>    >
>    >  output:
>    >        22-03-25 10:06:11.457 : [Validation] iter:5000, Average PSNR : 23.96dB
>    >        22-03-25 11:30:21.689 : [Validation] iter:10000, Average PSNR : 23.95dB
>    >        22-03-25 12:54:19.929 : [Validation] iter:15000, Average PSNR : 23.96dB
>    >        22-03-25 14:18:30.008 : [Validation] iter:20000, Average PSNR : 23.97dB
>    >        22-03-25 15:42:32.057 : [Validation] iter:25000, Average PSNR : 23.97dB
>    >        22-03-25 17:06:29.785 : [Validation] iter:30000, Average PSNR : 23.96dB
>    >        22-03-25 18:30:27.290 : [Validation] iter:35000, Average PSNR : 23.97dB
>    >        22-03-25 19:54:29.104 : [Validation] iter:40000, Average PSNR : 23.96dB
>    >        22-03-25 21:18:28.774 : [Validation] iter:45000, Average PSNR : 23.97dB
>    >        22-03-25 22:42:33.977 : [Validation] iter:50000, Average PSNR : 23.95dB
>    >        22-03-26 00:06:36.079 : [Validation] iter:55000, Average PSNR : 23.91dB
>    >        22-03-26 00:06:36.079 : [Validation] iter:55000, Average PSNR : 23.89dB
>    >  ```
>    >  j. 根据步骤k确定的iter, 进行 SwiniPassr++ 模型融合
>    >  ```
>    >  python swinipapssr_final.py --path ./superresolution/ssr_flickr1024_swinipapssr_plus --iter "步骤k确定的iter"
>    >  ```
>    >  k. 可以在 "./pretrained" 中查看，是否存在 "swinipapssr_final.pth" 文件，如果存在则继续执行，否则重复步骤j
>    >  l. 移动并重命名 SwiniPapssr++ 模型
>    >  ```
>    >  mv ./pretrained/swinipapssr_final.pth ./final_models/P24W12D9E180H9P.pth
>    >  ```
>    >