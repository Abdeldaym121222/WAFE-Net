Wavelet-Attention Feature Enhancement Network for Facial Expression Recognition
  
Facial expression recognition (FER) is still a difficult problem in affective computing due to the intra-class variability issue, different head poses, and lighting conditions. Current deep-learning methods for FER have only considered spatial features extracted through convolutional layers without taking into account the complementary frequency-based features that encode texture patterns generated through muscular movements. As a solution to this limitation, the WAFE-Net, Wavelet-Attention Feature Enhancement Network, has been proposed to incorporate both spatial and frequency domains using a spatial convolutional network along with a parallel wavelet transform module (WTM). The spatial path includes a pre-trained EfficientNet-B3 backbone, extended by a Dual Residual Attention Block (DRAB), which consists of dual attention layers. On the other hand, the wavelet path uses a 2D Haar discrete wavelet transform for decomposition of the feature maps into four frequency bands, which include high-frequency bands capturing fine textural information such as wrinkles, skin texture, and muscle movements. Finally, both branches are combined using the cross-domain fusion (CDF) block. Experiments using three benchmark databases, FER2013, KDEF, and CK+, show the effectiveness of WAFE-Net by attaining an accuracy of 72.89%, 99.49%, and 99.90%, respectively. The network is relatively lightweight compared to Transformer- and multi-branch attention-based FER models, with 22.4 million parameters and 1.55 GFLOPs, enabling faster computations with an estimated frame rate of 28.2 frames per second. Studies show how each component in the architecture contributes to the model, ensuring that WAFE-Net performs well in recognizing real-life facial expressions.

<img width="502" height="334" alt="image" src="https://github.com/user-attachments/assets/18f1b896-2014-44b7-b955-957f54ffb953" />



Figure 1: WAFE-Net: Overall Architecture


