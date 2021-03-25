This repo contains code for "Membership Inference Attacks and Defenses in Classification Models". This repo is tested using the following libraries: 

- Python 3.6
- Pytorch 1.2.0
- Scipy 1.5.2
- Scikit-learn 0.23.1
- Numpy 1.18.4

GPU support is highly recommended since multiple models need to be trained.	

------

### Code

Data subdirectory contains code to process the data into numpy array and divide data into disjoint parts for training/testing/membership evaluation. 

Model subdirectory contains code for implementation of different models.

Utils subdirectory contains code for some utility functions.

Different attacks are implemented in blackbox_attack.py.

Attack_exp.py contains code for model training and membership inferecen attack evaluation.

Attack_exp.sh is used to run the experiment and there are a few arguments that need to be figured out.

-----

### Citation

If you use this code, please cite the following paper:

```
@article{li2020membership,
  title={Membership inference attacks and defenses in supervised learning via generalization gap},
  author={Li, Jiacheng and Li, Ninghui and Ribeiro, Bruno},
  journal={arXiv preprint arXiv:2002.12062},
  year={2020}
}
```

This paper is going to be presented in CODASPY 2021 and this will be updated later.

-----

### Notes

Notes for datasets used in this paper: due to the file size limit, we are not able to provide you the datasets we use but basically we process each dataset into 4 different numpy arrays: train_data, train_labels, test_data and test_labels. After this, we load these 4 numpy arrays and process the data into disjoint parts for training/testing/membership inference evaluation. I can share the data upon request.

Notes for DP-SGD experiments: DP-SGD currently only supports alexnet and vgg-16. if you want to use ResNet or DenseNet, then the BatchNorm layer needs to be converted to GroupNorm layer using the coverting function in the attack_exp.py. You may also want to check out the opacus library for more details. If the converted model is used, the total number of training epochs should be 300 because the GroupNorm layer will slow down the convergence rate.



If you have any questions, please feel free to email me at li2829@purdue.edu.

