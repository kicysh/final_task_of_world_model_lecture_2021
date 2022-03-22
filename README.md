# final_task_of_world_model_lecture_2021

## 論文
### models
[scVI](https://www.nature.com/articles/s41592-018-0229-2.epdf?author_access_token=5sMbnZl1iBFitATlpKkddtRgN0jAjWel9jnR3ZoTv0P1-tTjoP-mBfrGiMqpQx63aBtxToJssRfpqQ482otMbBw2GIGGeinWV4cULBLPg4L4DpCg92dEtoMaB1crCRDG7DgtNrM_1j17VfvHfoy1cQ%3D%3D)

[ZIFA: Dimensionality reduction for zero-inflated single-cell gene expression analysis](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-015-0805-z)  
zero-inflated、つまり実際にはあるはずだが観測値がzeroとなっているデータを表現するためのレイヤーの実装。　

[DLVAE](https://academic.oup.com/bioinformatics/article/36/11/3418/5807606#supplementary-data) (library: [sorce](https://github.com/tabdelaal/scVI/tree/b20e34f02a87d16790dbacc95b2ae1714c08615c))  
今回の実装をする。scVIのうち、decoderを変化させている。

### data
[Pijuan-SalaB.et al.  (2019) A single-cell molecular map of mouse gastrulation and early organogenesis. Nature, 566, 490–495]  

### 前処理
[Seurat v3](https://www.cell.com/cell/fulltext/S0092-8674(19)30559-8)([home page](https://satijalab.org/seurat/))  
Rでのパッケージを実装してある。最新はv4。  
pythonにおいては scanpy.pp.highly_variable_genes で実装済み。
