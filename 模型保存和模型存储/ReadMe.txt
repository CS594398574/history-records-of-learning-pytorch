1、state_dict:
在pytorch中，torch.nn.模块模型的可学习参数（即权重和偏差）包含在模型的参数中（通过model.parameters（）访问）。
state-dict只是一个python字典对象，它将每个层映射到其参数张量。请注意，只有具有可学习参数的层（卷积层、线性层等）和注册缓冲区（batchnorm的running_mean）在模型的state_dict中有条目。
优化器对象（torch.optim）也有state_dict，其中包含优化器状态的信息以及使用的超参数。
由于state-dict对象是Python字典，因此可以轻松地保存、更新、更改和恢复它们，从而为pytorch模型和优化器添加了大量模块化。

2、Save/Load state_dict格式：
torch.save(model.state_dict(), PATH)
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
保存模型进行预测时，只需保存训练模型的学习参数。使用torch.save（）函数保存模型的状态“dict”将为您以后恢复模型提供最大的灵活性，这就是为什么它是保存模型的推荐方法。
一个常见的pytorch约定是使用.pt或.pth文件扩展名保存模型。
请记住，在运行预测之前，必须调用model.eval（）将Dropout和批处理规范化层设置为评估模式。不这样做将产生不一致的推理结果。
注意，load_state_dict（）函数采用字典对象，而不是保存对象的路径。这意味着您必须在将保存的state_dict传递给load_state_dict（）函数之前对其进行反序列化。
例如，不能使用model.load_state_dict（path）加载。



3、Save/Load Entire Model格式：
torch.save(model, PATH)
# Model class must be defined somewhere
model = torch.load(PATH)
model.eval()
这个保存/加载过程使用最直观的语法，所涉及的代码量最少。以这种方式保存模型将使用Python的pickle模块保存整个模块。
这种方法的缺点是，序列化数据绑定到特定的类和保存模型时使用的确切目录结构。原因是pickle不保存模型类本身。相反，它保存了一个包含类的文件的路径，该类在加载期间使用。
因此，当在其他项目中或在重构之后使用时，代码可以以各种方式中断。



4、#保存和加载用于预测或者恢复训练的通用的checkpoint.
保存：
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)

读取：
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()
# - or -
model.train()

注解：在保存通用checkpoint时候，用于预测或者恢复训练是，必须保存的不仅仅是model的state_dict.保存优化器的state_dict也很重要。
因为这些缓冲区和参数也会随着模型训练而更新。您可能希望保存的其他项目包括您未用的epoch、最新记录的训练loss、external torch.nn. Embedding层等
要保存多个组件，请将它们组织到字典中，然后使用torch.save（）对字典进行序列化。一个常见的pytorch约定是使用.tar文件扩展名保存这些检查点。
要加载这些项，首先初始化模型和优化器，然后使用torch.load（）在本地加载字典。从这里，您可以轻松地访问保存的项目，只需按预期查询字典即可。

请记住，在进行预测之前，必须调用model.eval（）将Dropout和批处理规范化层设置为评估模式。
不这样做将产生不一致的预测结果。如果您希望恢复训练，请调用model.train（）以确保这些层处于训练模式。

5、在一个文件中保存多个model
Save存储文件：
torch.save({
            'modelA_state_dict': modelA.state_dict(),
            'modelB_state_dict': modelB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            ...
            }, PATH)

读取文件：
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()

6、当一个模型包含多个torch.nn.Modules，比如Gan，序列到序列的模型，或者一个enseamble的model。可以用这种方法来save checkpoint
总之，save每个model的state_dict和相关优化器的state_dict的字典。可以用于恢复train
一个常见的pytorch约定是使用.tar文件扩展名保存这些检查点。

要加载模型，首先初始化模型和优化器，然后使用torch.load（）在本地加载字典。从这里，您可以轻松地访问保存的项目，只需按预期查询字典即可。
请记住，在预测之前，必须调用model.eval（）将Dropout和批处理规范化层设置为评估模式。不这样做将产生不一致的预测结果。
如果您希望恢复train，请调用model.train（）将这些层设置为训练模式。

7、使用不同模型中参数warmstarting模型
保存：torch.save(modelA.state_dict(), PATH)
读取：
modelB = TheModelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict=False)
注解：迁移学习或者训练一个新的复杂模型就需要加载部分模型或者部分加载模型，利用已经训练好的训练参数，甚至只有一些参数能用，这能预热训练过程，甚至
有助于加快你的模型从开始训练的收敛的速度。
无论是从缺少某些键值的部分state_dict中加载，还是使用比要加载到的模型更多的键加载状态state_dict,函数将strict参数设置为false用于忽略不匹配的键。
如果要将参数从一个层加载到另一个层，但某些键不匹配，只需在加载状态下更改参数键的名称，以匹配要加载到的模型中的键。

8、Save on GPU, Load on CPU
save：
    torch.save(model.state_dict(), PATH)
load：
    device = torch.device('cpu')
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH, map_location=device))

Save on GPU, Load on GPU
save：
    torch.save(model.state_dict(), PATH)
load：
    device = torch.device("cuda")
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.to(device)

Save on CPU, Load on GPU
Save：
    torch.save(model.state_dict(), PATH)
Load:
    device = torch.device("cuda")
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
    model.to(device)

9、Saving torch.nn.DataParallel Models
save:
torch.save(model.module.state_dict(), PATH)
load:
# Load to whatever device you want


torch.nn.dataparallel是一个支持并行GPU利用的模型包装器.
要一般保存数据并行模型，请保存model.module.state_dict（）。这样，您就可以灵活地以任何方式将模型加载到您想要的任何设备上。
