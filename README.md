# FARM: A Fast, Accurate, and Robust Process Monitoring Framework
Codebase for paper "Online Fault Detection and Classification of Chemical Process Systems Leveraging Statistical Process Control and Riemannian Geometric Analysis"

FARM is a fault detection and diagnosis framework designed for industrial processes. It has a holistic architechture as shown on figure below and it consists of two distinct modules, one for fault detection and the other for fault classification. In fault detection module, we adopt the state-of-the-art
eCDF-based SPC algorithm recently developed by Zheng et al. (2024). Moreover, the fault classification is carried out using the covariance matrices of the available time series data using support vector machine (SVM) method by considering the Riemannian geometry proposed by Smith et al (2022). This means that covariance matrices are mapped to the tangent space of the manifold at the mean, which enhances fault classification accuracy by 20%. 

The following files are released and are free of use. The Jupyter notebook `FARM` demonstrates how to use FARM.
-  All in-control and faulty simulation data are included in dataset/MATLAB GUI folder. These simulations are for the benchmark Tennessee Eastman Process (TEP) and they are carried out using the MATLAB GUI provided by Andersen et al (2022). The GUI code is available at: [TEP](https://github.com/dtuprodana/TEP)
-  `models.py `: includes the fault detection model based on the eCDF method.
-  `utils.py`: includes all utility functions used for easier data handling and computations.
We acknowledge the codes provided by the following developers that set the foundations of FARM:
- [Decentralized_monitoring](https://github.com/ZiqianZheng/Decentralized_monitoring)
- [RiemannianSPD](https://github.com/zavalab/ML/tree/master/RiemannianSPD)

## References:
- Emil B. Andersen, Isuru A. Udugama, Krist V. Gernaey, Abdul R. Khan, Christoph Bayer, and Murat
Kulahci. An easy to use gui for simulating big data using tennessee eastman process. Quality and
Reliability Engineering International, 38(1):264–282, 2022. doi: https://doi.org/10.1002/qre.2975.

- Alexander Smith, Benjamin Laubach, Ivan Castillo, and Victor M. Zavala. Data analysis using
riemannian geometry and applications to chemical engineering. Computers & Chemical Engineering,
168:108023, 2022

- Zheng, Ziqian, Jiahui Zhang, Lingyun Xiao, Warren R. Williams, Jing-Ru C. Cheng, and Kaibo Liu. 2024. Online Nonparametric Process Monitoring for IoT Systems Using Edge Computing. IISE Transactions, June, 1–15. doi:10.1080/24725854.2024.2352578.

