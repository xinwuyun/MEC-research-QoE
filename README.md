# <center>程序包使用说明</center>

本说明包括：

- 文件组成、内容介绍及使用方法
- 程序参数如何修改
- 程序修改参数后如何运行得到新的结果

## 1.文件组成及概述

以程序包的`python实现`文件夹为根目录，其中文件目录如下：

<img src="https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220411170631776.png" alt="image-20220411170631776" style="zoom:50%;" />

<center><strong>图1：程序包目录结构</strong></center>

### 1.1 数据文件

#### 1.1.1 原始数据`./data`

原始数据指用于生成站点的服务范围$R_j$，容量$C_j$，用户移动速度$speed$，初始运动角度$\theta$，初始$QoS$等级的原始数据文件，在目录`python实现/data`目录下，包括`site-optus-melbCBD.csv`和`users-melbcbd-generated.csv`。

<img src="https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220410200232445.png" alt="image-20220410200232445" style="zoom:67%;" />

<center><strong>图2：data文件夹目录结构</strong></center>

- 文件1`site-optus-melbCBD.csv`：

  来源：`eua-dataset-master\edge-servers\site-optus-melbCBD.csv`

  **墨尔本CBD的MEC服务站信息，主要使用其经纬度信息。**

- 文件2`users-melbcbd-generated.csv`

  来源：`eua-dataset-master\users\users-melbcbd-generated.csv`

  **用户的位置信息。**

后续程序的运行必须的数据文件即此两个文件。若需要**替换这两个文件为其他的站点/用户，新文件的数据模式必须包含**：

- 文件1：MEC的**服务站经纬度**，其他的如NAME等列都是不必要的

  <img src="https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220410191208941.png" alt="image-20220410191208941" style="zoom: 50%;" />

  <center><strong>图3：服务站基本信息要求</strong></center>

- 文件2：**用户的经纬度**

  <img src="https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220410192435622.png" alt="image-20220410192435622" style="zoom:50%;" />

  <center><strong>图4：用户基本信息要求</strong></center>

#### 1.1.2 输入数据

data目录下的其他文件为基于上述两个文件通过**随机算法（1.2.1中说明）**产生的mobMig算法输入数据集，分别是：

- `./data/server_la_lng_r200-400_capicity200.csv`

  该csv文件为主算法`mobMig_QoE.py`的输入数据文件，提供站点的所有相关信息。

  该文件在`site-optus-melbCBD.csv`的基础上使用随机算法`servers_info_generator.ipynb`产生。随机算法产生额外信息包括站点标号`SITE_ID`，服务半径`r`，以及最大可服务人数`capacity`，表的属性如下：

  |      | `SITE_ID`   | `LATITUDE` | `LONTITUDE` | `r`                        | `capacity`                  |
  | ---- | ----------- | ---------- | ----------- | -------------------------- | --------------------------- |
  | 含义 | 站点的标号  | 纬度       | 经度        | 信号范围（服务范围）       | 最大可服务人数              |
  | 值域 | 从1开始整数 | `-`        | `-`         | $200\sim 400 $范围内随机数 | $7$(可随机更改为其他合理值) |

  ![image-20220410194018149](https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220410194018149.png)

  <center><strong>图5：经过随机算法后产生的完整服务器信息</strong></center>

- `./data/行人_id_速度_角度.csv`

  该csv文件为主算法`mobMig_QoE.py`的输入数据文件，提供用户的所有相关信息。

  该文件再在`users-melbcbd-generated.csv`基础上使用随机算法`user_info_generator.ipynb`产生。随机算法产生额外信息包括站点的标号`USER_ID`，运动方向`theta`，速度大小`v`，用户初始最高QoS等级`H`，用户初始最低QoS等级`L`，表的属性如下：

  |      | `USER_ID`   | `LATITUDE` | `LONGTITUDE` | `theta`            | `v`                                                          | `H`                       | `L`                       |
  | ---- | ----------- | ---------- | ------------ | ------------------ | ------------------------------------------------------------ | ------------------------- | ------------------------- |
  | 含义 | 用户的标号  | 纬度       | 经度         | 用户的初始运动角度 | 速度大小                                                     | 初始最高QoS等级           | 初始最低QoS等级           |
  | 值域 | 从1开始整数 | `-`        | `-`          | $(0,2\pi)$         | 车辆：占20%，范围为2.68224m/s~10.72896m/s<br />行人：占80%，范围为0.44704m/s~1.34112m/s | 1或2或3（等级范围可扩大） | 1或2或3（等级范围可扩大） |

  ![image-20220410195019874](https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220410195019874.png)

  <center><strong>图6：经过随机算法产生的完整用户信息</strong></center>



#### 1.1.3 输出数据`./result`

输出数据分为以下两类：
$$
数据结果\begin{cases}mogMig.py的运行结果\begin{cases}50\_users\_result.csv：用户数量为50人时的各项指标结果\\
100\_users\_result.csv：用户数量为100人时的各项指标结果\\
150\_users\_result.csv：用户数量为150人时的各项指标结果\\
200\_users\_result.csv：用户数量为200人时的各项指标结果\end{cases}\\
用于对比算法性能的最终结果文件：average\_result.csv\end{cases}
$$
目录结果如下，下面对两类结果进行说明。

<img src="https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220411172036054.png" alt="image-20220411172036054" style="zoom:50%;" />

<center><strong>图7：主程序运行结果result的目录结构</strong></center>

##### （1）主程序mobMig_QoE输出数据文件

输出数据有五个文件，上述的四个文件是主算法`mobMig_QoE.py`的运行结果。四个文件分别对应用户人数为50,100,150,200人时的结果

以50人时的结果`50_users_result.csv`为例，说明输出数据的信息：

<img src="https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220410200556758.png" alt="image-20220410200556758" style="zoom:50%;" />

<center><strong>图8：输出结果信息</strong></center>

- 时刻

  每隔5s计算一次其他四个属性的值（每25s为QoS随机改变的时刻）

- 用户覆盖率：

  计算公式为：$用户覆盖率 = \frac{得到服务的用户数}{总用户数}$

  在本表中，总用户数为随机抽取的50人

- 重分配次数

  在MobMig算法的`mobility_aware_migration`过程中会根据用户的移动状态对其进行更优重分配

  若一个用户由站点A在该过程中被分配给站点B，那么重分配数自增1

- 已分配用户数  Number of Allocated Users

  已经得到服务的用户数

- QoE值 QoE value

  所有用户QoE值，将时刻为25的整数倍的QoE值提取，除以20得到所需的指标Average QoE value

##### （2）用于对比算法性能的最终结果文件

数据处理文件中的结果即是下图中需要的针对mobmig算法的不同用户数量时的各值指标。

![image-20220411171222344](https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220411171222344.png)

表格内容如下，含义直接，不再赘述

![image-20220411172404038](https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220411172404038.png)

<center><strong>图9：对比算法需要最终数据</strong></center>

### 1.2 程序文件

程序文件包含数据预处理的两个`.ipynb`文件和主程序`mobMig_QoE.py`。

#### 1.2.1 数据预处理程序

数据预处理程序包含用户数据的预处理和站点数据的预处理，生成除了经纬度信息外的其他信息。

- `user_info_generator.ipynb`

  输入数据：用户的包括经度和纬度在内的位置信息，$用户信息 = \{LATITUDE, LONGTITUDE\}$

  输出数据：随机函数处理后的数据，$用户信息 =\{USER\_ID, LATITUDE, LONGTITUDE, theta, v, H, L\}$

- `servers_info_generator.ipynb`

  输入数据：输入数据集要求有MEC服务器的经纬度信息，即$MEC服务器信息 = \{LATITUDE, LONGITUDE\}$

  输出数据：随机函数处理后的数据，$MEC服务器信息 = \{SITE\_ID, LATITUDE, LONGITUDE, r, capacity\}$

#### 1.2.2 主程序文件

主程序文件为`mobMig_QoE.ipynb`将预处理后的数据文件，即1.1.2 中的输入数据作为程序的输入，输出结果存于1.1.3 中的输出数据文件夹中。

## 2.程序的运行

可运行程序中包含一些可修改的参数以及必须要修改以多次运行得到结果的参数：
$$
可修改参数\begin{cases}预处理程序中\begin{cases}servers\_info\_generator.ipynb\begin{cases}服务器服务范围\\服务容量\end{cases}\\user\_info\_gnerator.ipynb\begin{cases}用户中汽车与行人的比例\\用户初始速度\\用户初始运动角度\\用户初始QoS等级\end{cases}\end{cases}\\主程序mogMig\_QoE.py中\begin{cases}迭代时间间隔：不建议修改\\
用户数量：每次运行为了得到不同用户数的结果，必须手动修改为50,100,150,200四个值\end{cases}\end{cases}
$$
其中**用户数量**要得到完整的4个文件结果**必须修改四次**。**其他参数都可以不修改，直接运行即可。**详细说明如下：

### 2.1 数据预处理程序

两个预处理程序包括`user_info_generator.ipynb`和`servers_info_generator.ipynb`。

两个数据预处理文件都只需要对应对象的经纬度信息，产生除了位置信息外的其余的信息。产生的信息说明在**1.1.2**中已经说明，这里不做赘述。下面对如何**修改用户和MEC服务器的相关信息的参数**进行说明。（这部分都可以不修改）

#### 2.1.1 服务器的参数的修改

##### （1）服务器的服务范围Rj

按照论文`Panda2020_Chapter_DynamicEdgeUserAllocationWithU.pdf`中描述的实验设置，将服务范围设置为200~400之间的随机数。数值可在`server_info_generator.ipynb`文件中修改：

<img src="https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220410203026940.png" alt="image-20220410203026940" style="zoom:50%;" />

<center><strong>图10：对服务范围和服务人数两参数的修改</strong></center>

##### （2）服务人数$capacity$

服务人数根据论文`MobMig.pdf`中容量进行设置，论文`Panda2020.....`中未作具体要求。这里设置为7人，数值可以在`server_info_generator.ipynb`文件中修改。参见图9：对服务范围和服务人数两参数的修改。

#### 2.1.2 用户参数的修改

##### （1）用户比例的修改

用户比例为80%的行人与20%的车辆，二者具有不同的速度范围。若要更改车辆和行人的比例，可在`user_info_gennerator.ipynb`文件中如下图处修改：

<img src="https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220410204606335.png" alt="image-20220410204606335" style="zoom:50%;" />

<center><strong>图11：对用户中车辆和行人的比例的修改</strong></center>

##### （2）用户的初始运动角度

初始运动角度为$(0, 2\pi)$，该参数用于随机漫步使用，可以修改但没必要。

<img src="https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220410203909241.png" alt="image-20220410203909241" style="zoom:50%;" />

<center><strong>图12：对用户初始运动角度的修改</strong></center>

##### （3）速度大小

速度大小设置的范围为按照论文`Panda2020....`中实验要求进行设置

- 行人 0.44704m/s~1.34112m/s

- 车辆 2.68224m/s~10.72896m/s

若修改可在`user_info_gennerator.ipynb`文件中如下图处修改：

<img src="https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220410204046276.png" alt="image-20220410204046276" style="zoom:50%;" />

<center><strong>图13：对用户初始速度大小的修改</strong></center>

##### （4）初始QoS等级

QoS等级按照论文`Panda2020...`进行设置，与论文中相同，设置了三个等级，每个等级的向量的四个纬度设置也相同。若要修改需要修改两处：

- `user_info_generator.ipynb`

  在该文件下修改用户的最低等级和最高等级

  <img src="https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220410205020160.png" alt="image-20220410205020160" style="zoom:50%;" />

  <center><strong>图14：对用户QoS等级数的修改</strong></center>

- `mobMig_QoE.py`

  在该文件下修改等级的具体的向量的四个维度信息或者是新增/减少向量

  <img src="https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220410205332319.png" alt="image-20220410205332319" style="zoom:50%;" />

  <center><strong>图15：对用户QoS向量的修改</strong></center>

### 2.2 主程序`mobMig_QoE.py`——运行4次得到结果

#### 2.2.1 user_number的修改

`mobMig`中需要修改，以多次运行得到结果的参数是用户的数量user_number，该值在50到200间以50为单位变化，若要修改user_number，在主程序的430行，如图15修改即可。**改为50,100,150,200后可分别运行，运行结果存储在result目录下以50,100,150,200开头的xx_users_result.csv文件中**，详见图7：主程序运行结果result的目录结构

<img src="https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220410205848183.png" alt="image-20220410205848183" style="zoom:67%;" />

<center><strong>图16：对用户数量的修改</strong></center>

#### 2.2.2 对迭代时间的修改

迭代时间不建议修改，按照5s改变随机漫步的方向较为合适。若修改，那么修改为25的因数：1,5,25，否则程序出现问题。

<img src="https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220410210431358.png" alt="image-20220410210431358" style="zoom: 80%;" />

<center><strong>图17：对迭代时间间隔的修改</strong></center>

### 2.3 结果数据处理程序

数据处理程序按照论文`Panda2020...`实验设置中的描述，分别对50个用户，100个用户，150个用户和200个用户时的结果进行平均值计算。按照论文的处理，将25s整数倍数时的数据进行提取，分别求其平均值，得到不同用户数量时各个指标的结果。

该程序若可能有需要调整的参数，那么除了提取数据的间隔：25s外无需要修改的地方。若需要修改此间隔，在结果数据处理程序`result_handle.ipynb`中如下图处修改变量`time_slot`为其他值即可

<img src="https://cdn.jsdelivr.net/gh/Holmes233666/gitee-image@main/pictureStore/image-20220411170201210.png" alt="image-20220411170201210" style="zoom:50%;" />

<center><strong>图18：对结果数据提取的时间间隔的修改</strong></center>

### 2.4 运行顺序

若更改了数据源data目录下的基本数据，或者修改数据预处理程序的相关参数，那么算法需要完整重新运行一遍才能得到新的最终结果，运行顺序为：
