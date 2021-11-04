# python异常捕获与处理

### 1. 异常简介
* 异常不是错误

  错误通常指的是语法错误，可以人为避免。
  异常是指在语法逻辑正确的而前提下，出现的问题。

* 异常即是一个事件
  该事件会在程序执行过程中发生，影响了程序的正常执行。一般情况下，在Python无法正常处理程序时就会发生一个异常。异常是Python对象，表示一个错误。当Python脚本发生异常时我们需要捕获处理它，否则程序会终止执行。

当一个未捕获的异常发生时，Python将结束程序并打印一个堆栈跟踪信息，以及异常名和附加信息。具体如下：
```bash
Traceback (most recent call last): 
File "<ipython-input-1-0bc85e309fb1>", line 1, in <module> 
    min(x,y) 
NameError: name 'x' is not defined
```

* 异常捕获主要目的
  * 错误处理：在运行时出现错误的情况下，应用程序可能会无条件终止。使用异常处理，我们可以处理失败的情况并避免程序终止。

  * 代码分离：错误处理可以帮助我们将错误处理所需的代码与主逻辑分离。与错误相关的代码可以放置在“ except ”块中，该块将其与包含应用程序逻辑的常规代码隔离开来。

  * 错误区分：帮助我们隔离执行过程中遇到的不同类型的错误。我们可以有多个“ except”块，每个块处理一种特定类型的错误。

* 异常捕获其他应用
  * 事件通知：异常也可以作为某种条件的信号，而不需要在程序里传送结果标志或显式地测试它们。

  * 特殊情形处理：有时有些情况是很少发生的，把相应的处理代码改为异常处理会更好一些。

  * 特殊的控制流：异常是一个高层次的”goto”，可以把它作为实现特殊控制流的基础。如反向跟踪等。

Python自带的异常处理机制非常强大，提供了很多内置异常类，可向用户准确反馈出错信息。由于Python是面向对象语言，认为一切皆对象，所以异常也是对象。Python异常处理机制中的BaseException是所有内置异常的基类，但用户定义的类并不直接继承BaseException，所有的异常类都是从Exception继承，且都在exceptions模块中定义。

但是，因为Python自动将所有异常名称放在内建命名空间中，所以程序不必导入特定模块即可使用异常。

* Python内置异常类继承层次结构
```bash
-- SystemExit  # 解释器请求退出
-- KeyboardInterrupt  # 用户中断执行(通常是输入^C)
-- GeneratorExit  # 生成器(generator)发生异常来通知退出
     -- StopIteration  # 迭代器没有更多的值
     -- StopAsyncIteration  # 必须通过异步迭代器对象的__anext__()方法引发以停止迭代
     -- ArithmeticError  # 各种算术错误引发的内置异常的基类
     |    -- FloatingPointError  # 浮点计算错误
     |    -- OverflowError  # 数值运算结果太大无法表示
     |    -- ZeroDivisionError  # 除(或取模)零 (所有数据类型)
     -- AssertionError  # 当assert语句失败时引发
     -- AttributeError  # 属性引用或赋值失败
     -- BufferError  # 无法执行与缓冲区相关的操作时引发
     -- EOFError  # 当input()函数在没有读取任何数据的情况下达到文件结束条件(EOF)时引发
     -- ImportError  # 导入模块/对象失败
     |    -- ModuleNotFoundError  # 无法找到模块或在在sys.modules中找到None
     -- LookupError  # 映射或序列上使用的键或索引无效时引发的异常的基类
     |    -- IndexError  # 序列中没有此索引(index)
     |    -- KeyError  # 映射中没有这个键
     -- MemoryError  # 内存溢出错误(对于Python 解释器不是致命的)
     -- NameError  # 未声明/初始化对象 (没有属性)
     |    -- UnboundLocalError  # 访问未初始化的本地变量
     -- OSError  # 操作系统错误，EnvironmentError，IOError，WindowsError，socket.error，select.error和mmap.error已合并到OSError中，构造函数可能返回子类
     |    -- BlockingIOError  # 操作将阻塞对象(e.g. socket)设置为非阻塞操作
     |    -- ChildProcessError  # 在子进程上的操作失败
     |    -- ConnectionError  # 与连接相关的异常的基类
     |    |    -- BrokenPipeError  # 另一端关闭时尝试写入管道或试图在已关闭写入的套接字上写入
     |    |    -- ConnectionAbortedError  # 连接尝试被对等方中止
     |    |    -- ConnectionRefusedError  # 连接尝试被对等方拒绝
     |    |    -- ConnectionResetError    # 连接由对等方重置
     |    -- FileExistsError  # 创建已存在的文件或目录
     |    -- FileNotFoundError  # 请求不存在的文件或目录
     |    -- InterruptedError  # 系统调用被输入信号中断
     |    -- IsADirectoryError  # 在目录上请求文件操作(例如 os.remove())
     |    -- NotADirectoryError  # 在不是目录的事物上请求目录操作(例如 os.listdir())
     |    -- PermissionError  # 尝试在没有足够访问权限的情况下运行操作
     |    -- ProcessLookupError  # 给定进程不存在
     |    -- TimeoutError  # 系统函数在系统级别超时
     -- ReferenceError  # weakref.proxy()函数创建的弱引用试图访问已经垃圾回收了的对象
     -- RuntimeError  # 在检测到不属于任何其他类别的错误时触发
     |    -- NotImplementedError  # 在用户定义的基类中，抽象方法要求派生类重写该方法或者正在开发的类指示仍然需要添加实际实现
     |    -- RecursionError  # 解释器检测到超出最大递归深度
     -- SyntaxError  # Python 语法错误
     |    -- IndentationError  # 缩进错误
     |         -- TabError  # Tab和空格混用
     -- SystemError  # 解释器发现内部错误
     -- TypeError  # 操作或函数应用于不适当类型的对象
     -- ValueError  # 操作或函数接收到具有正确类型但值不合适的参数
     |    -- UnicodeError  # 发生与Unicode相关的编码或解码错误
     |         -- UnicodeDecodeError  # Unicode解码错误
     |         -- UnicodeEncodeError  # Unicode编码错误
     |         -- UnicodeTranslateError  # Unicode转码错误
     -- Warning  # 警告的基类
          -- DeprecationWarning  # 有关已弃用功能的警告的基类
          -- PendingDeprecationWarning  # 有关不推荐使用功能的警告的基类
          -- RuntimeWarning  # 有关可疑的运行时行为的警告的基类
          -- SyntaxWarning  # 关于可疑语法警告的基类
          -- UserWarning  # 用户代码生成警告的基类
          -- FutureWarning  # 有关已弃用功能的警告的基类
          -- ImportWarning  # 关于模块导入时可能出错的警告的基类
          -- UnicodeWarning  # 与Unicode相关的警告的基类
          -- BytesWarning  # 与bytes和bytearray相关的警告的基类
          -- ResourceWarning  # 与资源使用相关的警告的基类。被默认警告过滤器忽略。
```

### 2. 异常捕获与处理

```bash
try:
    执行代码
except [ (Error1, Error2, ... ) [as e] ]:
    发生指定error1，2时执行的代码
except [ (Error3, Error4, ... ) [as e] ]:
    发生指定error3，4时执行的代码
except  [Exception]:
    未发生指定error时执行的代码
else:
    没有异常时执行的代码块
finally:
    不管有没有异常都会执行的代码块
```

如果当try后的语句执行时发生异常，Python就从except中提到的指定error中寻找异常处理方式，如果没有指定的就从未指定的异常处理方式中寻找处理方式，异常处理完毕，控制流就通过整个try语句。如果在try子句执行时没有发生异常，Python将执行else语句后的语句（如果有else的话），然后控制流通过整个try语句。finally是一定执行的代码块儿。

* 代码实例：
```bash
for j in range(len(dir2)):#这里是打开一个文件夹读取里面很多文档信息，但是文档编码方式不同，故而通过这种方式顺利读取文档信息。
            # with open('Data/{0}/{1}'.format(dir1[i],dir2[j]),encoding='ISO-8859-1') as f:
            #     wordList = f.read().strip().replace('','')
            try:
                with open('Data/{0}/{1}'.format(dir1[i],dir2[j]),encoding='gbk') as f:
                    wordList = f.read()
            except:
                try:
                    with open('Data/{0}/{1}'.format(dir1[i], dir2[j]), encoding='utf-8') as f:
                        wordList = f.read()
                except:
                    with open('Data/{0}/{1}'.format(dir1[i], dir2[j]), encoding='gb18030') as f:
                        wordList = f.read().strip()
            corpus.append(wordList)
```

### 3. raise用法
* raise语法格式

```bash
raise [exceptionName[(reason)]]
```

等价于

```bash
raise  # 该语句引发当前上下文中捕获的异常（比如在 except 块中），或默认引发 RuntimeError 异常。
raise exceptionName  # 表示引发执行类型的异常。
raise exceptionName(reason)   # 在引发指定类型的异常的同时，附带异常的描述信息。
```

举例1：无参数raise
```bash
>>> raise
Traceback (most recent call last):
  File "<pyshell#0>", line 1, in <module>
    raise
RuntimeError: No active exception to reraise
```
举例2：raise exceptionName
```bash
>>> raise ZeroDivisionError
Traceback (most recent call last):
  File "<pyshell#1>", line 1, in <module>
    raise ZeroDivisionError
ZeroDivisionError
```
举例3：raise exceptionName（reason）：
```bash
>>> raise ZeroDivisionError('除数不能为零')
Traceback (most recent call last):
  File "<pyshell#2>", line 1, in <module>
    raise ZeroDivisionError('除数不能为零')
ZeroDivisionError: 除数不能为零
```

* 配合try主动引发异常
raise 语句引发的异常通常用 try except（else finally）异常处理结构来捕获并进行处理。
使用 raise 语句引发异常，程序的执行是正常的，手动抛出的异常并不会导致程序崩溃。
  * 示例：
  ```bash
  try:
    num1 = int(input('输入一个被除数 num1：'))  # 用户输入一个被除数
    num2 = int(input('输入一个除数 num2：'))  # 用户输入一个除数
    result = num1 / num2
    # 判断用户输入的除数是否为零
    if (num2 == 0):
        raise ZeroDivisionError
except ZeroDivisionError as e:
    print('引发异常：', repr(e))

```
* 运行结果：
```bash
输入一个被除数 num1：6
输入一个被除数 num2：0
引发异常： ZeroDivisionError('division by zero')
```
* 自定义异常
你可以通过创建一个新的异常类来拥有自己的异常。异常类继承自 Exception 类，可以直接继承，或者间接继承，例如：
```bash
class MyError(Exception):
   def __init__(self, value):
       self.value = value
       return repr(self.value)
   # raise MyError('oops!')
   print('My exception occurred, value:', e.value)
```

### 4. assert用法

设想一个情况，我们的代码中包含数据读取和数据处理的部分，其中数据处理需要GPU计算，而数据读取也需要大量时间，甚至读取数据之后还需要进一步的处理，如果我们在处理完之后才发现GPU不可用，就大大降低了效率，故而需要一种提前判断的方式去解决此问题。

因此我们提出断言assert！

Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况。


assert语法：
```bash
assert expression
```
等价于：
```bash
if not expression:
    raise AssertionError(arguments)
```
实例：
```bash
import torch
assert (torch.cuda.is_available()), "本机器需要有可用的GPU才能运行此代码！"
```
### 5.Traceback用法

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Citations

```bibtex
@
  title={Skip-Transformer Unet},
  author={Tianyu Yan, Fuzi Wan},
  journal={投哪一篇呢嘿嘿},
  year={2021}
}
```
