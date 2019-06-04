# IODINE

A Pytorch implementation of the paper [Multi-Object Representation Learning with Iterative Variational Inference](https://arxiv.org/abs/1903.00450). 

The model is trained and tested on the CLEVR6 and multi-dsprites datasets. 

## Results

**Decomposition**. The model learns to decompose objects into seperate images. Also, segmentation masks are learned. Results on CLEVR6:

![f1-clevr6-recons](pics/f1-clevr6-recons.png)

The following are reconstruction results on multi-dsprites. Note in this figure, the individual decomposed images are masked with the predicted masks. Note the model struggles to to learn sharp object boundaries. This is also mentioned in the original paper.

![f2-dsprites-recons](pics/f2-dsprites-recons.png)

For quantitative results, I measured ARI as in the paper. Results:

|                | this implementation | original paper |
| -------------- | ------------------- | -------------- |
| CLEVR6         | 0.971               | 0.988          |
| multi-dsprites | 0.740               | 0.767          |

There is a small gap between my implementation and results in the original paper. This might be due to minor differences in implementation.

**Disentanglement**. The following figure shows that the model learns to disentangle latent factors:

![f4-traversal](pics/f4-traversal.png)

During training, I found the factor "shape" is the most difficult to capture. But the model finally learned to do so. For example, the following latent traversal shows that the model captures the factor "shape":

![f5-traversal-shape](pics/f5-traversal-shape.png)

**Generalization**. The original paper reported that the model generalizes to scenes with more objects that seen during training. This is partially true in my experiments. The following are some decomposition results on the full CLEVR dataset:

![f3-clevr-recons](pics/f3-clevr-recons.png)

The above scenes are carefully selected by me. When I tested on the full CLEVR dataset, sometimes even though the number of objects is less then 6, the results can be pretty bad:

![f6-clevr-failure](pics/f6-clevr-failure.png)

I suspect that this is because I generated the CLEVR6 myself, and some rendering parameters that I chosen are different from those of the original CLEVR dataset.