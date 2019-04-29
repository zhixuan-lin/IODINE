# 4-29

Implemented some iterative inference models. Found that some factors that are especially important:

* Encode input, in addition to gradients
* Must use elu instead of relu. Possibly because the signs of the gradients 
  matter a lot.
* Using gated update form and highway connection in MLP is very helpful.

Encoding input vs. no input:

![F1](pics/F1.png)

Three lines are respectively: 

* ELU + gated update
* ELU
* No ELU

![F2](pics/F2.png)


