'''
torch.Tensor 是这个包的核心类。如果设置它的属性 .requires_grad 为 True，
那么它将会追踪对于该张量的所有操作。当完成计算后可以通过调用 .backward()，
来自动计算所有的梯度。这个张量的所有梯度将会自动累加到.grad属性。
注意：在 y.backward() 时，如果 y 是标量，则不需要为 backward() 传入任何参数；否则，需要传入一个与 y 同形的Tensor。
要阻止一个张量被跟踪历史，可以调用.detach()方法将其与计算历史分离，并阻止它未来的计算记录被跟踪。
为了防止跟踪历史记录(和使用内存），可以将代码块包装在 with torch.no_grad(): 中。在评估模型时特别有用，因为模型可能具有 
requires_grad = True 的可训练的参数，但是我们不需要在此过程中对他们进行梯度计算。
还有一个类对于autograd的实现非常重要：Function。Tensor 和 Function 互相连接生成了一个无环图 (acyclic graph)，它编码了完整的计算历史。
每个张量都有一个.grad_fn属性，该属性引用了创建 Tensor 自身的Function(除非这个张量是用户手动创建的，即这个张量的grad_fn是 None )。下面给出的例子中，张量由用户手动创建，因此grad_fn返回结果是None。
'''
