# CPN (ICCV2021) 
This is an implementation of [Complementary Patch for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2108.03852v1), which is accepted by ICCV2021 poster.

This implementation is based on [SEAM](https://github.com/YudeWang/SEAM) and [IRN](https://github.com/jiwoon-ahn/irn).
## Abstract 
Weakly Supervised Semantic Segmentation (WSSS) based on image-level labels has been greatly advanced by exploiting the outputs of Class Activation Map (CAM) to generate the pseudo labels for semantic segmentation. However, CAM merely discovers seeds from a small number of regions, which may be insufficient to serve as pseudo masks for semantic segmentation. In this paper, we formulate the expansion of object regions in CAM as an increase in information. From the perspective of information theory, we propose a novel Complementary Patch (CP) Representation and prove that the information of the sum of the CAMs by a pair of input images with complementary hidden (patched) parts, namely CP Pair, is greater than or equal to the information of the baseline CAM. Therefore, a CAM with more information related to object seeds can be obtained by narrowing down the gap between the sum of CAMs generated by the CP Pair and the original CAM. We propose a CP Network (CPN) implemented by a triplet network and three regularization functions. To further improve the quality of the CAMs, we propose a Pixel-Region Correlation Module (PRCM) to augment the contextual information by using object-region relations between the feature maps and the CAMs. Experimental results on the PASCAL VOC 2012 datasets show that our proposed method achieves a new state-of-the-art in WSSS, validating the effectiveness of our CP Representation and CPN.
## Prerequisite
* The requirements are in **requirements.txt**. However, the settings are not limited to it (CUDA 11.0, Pytorch 1.7 for one RTX3090). Besides,the batch size could be even
larger like 8 or 16 if you have sufficient GPU resources, which you may get higher performance than the paper reported.
* The pretrained_weight for the initialization of ResNet38 and well-trained CPN is [here](https://pan.baidu.com/s/1FcRHsLqxl3e2siKl0OCgEQ) 
 in BaiDuYun, and the code is **y6h4**, or you could find them in Google Drive, which is [here](https://drive.google.com/file/d/1tH6caIx1y0sGXSPneLTn2H0_qrtjYxIs/view?usp=sharing).
* PASCAL VOC 2012 devkit with expanded version, which includes 10582 training samples.
## Usage
1. Train the CPN to obtain the weight, which will be saved in "CPN/CPN". Remember to set the **VOC12 and pre-trained weight** path in the script. 
    ```
    python train_cpn.py
    ```

2. Generate the foreground seeds of CAM (without background) using the weight or the well-trained CPN, the results is in **out_cam**.

    ``` 
    python infer_cam.py 
    ```

3. Evaluate the CAM by selecting the background. Remember to set the data path of **VOC** in this script.
    ``` 
    python evaluation_cam.py
    ```
## Implementation of results in paper
1. I suggest to use the [IRN](https://github.com/jiwoon-ahn/irn) and the for the second expansion of the CAM. Although you can directly use 
the old version of [AffinityNet](https://github.com/jiwoon-ahn/psa), you may take long time to find the parameters to generate the CAM that reaches 
the reported performance. You can directly use the **well-trained weights from IRN** to generated the mask for segmentation.
2. For the segmentation model, we use the [DeepLab](https://github.com/YudeWang/semantic-segmentation-codebase/tree/main/experiment/seamv1-pseudovoc) here.


## Acknowledgement
Great thanks to the code of the [SEAM](https://github.com/YudeWang/SEAM) and [IRN](https://github.com/jiwoon-ahn/irn).

