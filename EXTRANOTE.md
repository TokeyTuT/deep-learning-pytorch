# YOLO 数据集 与 COCO 数据集



在计算机视觉中，有两种数据集格式经常被使用，COCO 格式和 YOLO 格式，下面我们来介绍一下这两种数据集的形式以及如何导入。



COCO 和 YOLO 两种格式最本质的区别在于：**文件组织方式**、**坐标原点/基准**以及**数值类型（像素绝对值 vs 归一化比例）**。

**核心概念对比**

| 对比维度     | COCO 格式 (`.json`)                                          | YOLO 格式 (`.txt`)                                           |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **存储方式** | **单个集中式大文件**：所有图片的标注、类别全塞在 `train.json` 内部 | **分布式单图单文件**：每张图片对应一个同名的 `.txt`（如 `001.jpg` 对应 `001.txt`） |
| **类别表示** | 键值对映射，ID 往往是任意数字（如 `category_id: 18`）        | 单个整数，必须从 `0` 开始连续递增（如 `0, 1, 2...`）         |
| **坐标基准** | **左上角坐标 + 宽高**：`[x_min, y_min, width, height]`       | **中心点坐标 + 宽高**：`[x_center, y_center, width, height]` |
| **数值单位** | **绝对像素值**（如 `100, 200`）                              | **相对归一化比例**（除以图片宽高后的 `0.0 ~ 1.0` 小数）      |

**具体计算示例**

假设有一张图片 **`cat_dog.jpg`**：

-   图片尺寸：**宽 W=1000 px，高 H=500 px**
-   类别字典：`0: 'cat'`, `1: 'dog'`
-   图中有一只猫（左上角目标）：
    -   左上角像素坐标：xmin=100,ymin=50
    -   目标框的像素尺寸：宽 w=200, 高 h=100

Plaintext

```
(0, 0)
┌──────────────────────────────────────── (W=1000)
│  (100, 50) 
│      ┌────────────┐ (200px 宽)
│      │  [猫 / cat] │
│      │   • 中心点  │ (100px 高)
│      └────────────┘
│
└──────────────────────────────────────── (H=500)
```

### **1. COCO 格式的表示方式（JSON）**

COCO 记录绝对像素值，直接记录左上角和长宽：

JSON

```
{
  "images": [
    { "id": 1, "file_name": "cat_dog.jpg", "width": 1000, "height": 500 }
  ],
  "categories": [
    { "id": 1, "name": "cat" },
    { "id": 2, "name": "dog" }
  ],
  "annotations": [
    {
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 50, 200, 100]  // [x_min, y_min, w, h] 绝对像素
    }
  ]
}
```



**对于 COCO** 数据集，它的每个键对应的 id 有如下解释：

其实你只要把 COCO 的 JSON 文件当成一个**关系型数据库**（或者多张 Excel 表），这些 ID 就瞬间清晰了。

COCO 里面主要有三大列表（三张表）：**`images`（图片表）**、**`categories`（类别字典表）** 和 **`annotations`（标注框表）**。

**一张图看懂它们的关系**

Plaintext

```
【图片表 images】              【类别表 categories】
 ├── id: 101 (图片身份证)        ├── id: 1 (猫)
 └── id: 102 (图片身份证)        └── id: 2 (狗)
         │                              │
         └──────────────┬───────────────┘
                        ▼
            【标注框表 annotations】
             ├── id: 1001 (框的唯一编号)
             │   ├── image_id: 101  ──────────> 这个框画在 101 号图片上
             │   ├── category_id: 2 ──────────> 这个框圈出来的是 2 号类别 (狗)
             │   └── bbox: [x, y, w, h]
             │
             └── id: 1002 (另一个框的编号)
                 ├── image_id: 101  ──────────> 也在 101 号图片上 (一图多目标)
                 ├── category_id: 1 ──────────> 这个框圈出来的是 1 号类别 (猫)
                 └── bbox: [x, y, w, h]
```

#### **4 个核心 ID 逐个拆解**

-   **`image['id']`（图片 ID / 主键）**
    -   **作用**：每张图片的**唯一身份证号**。
    -   整个 JSON 里绝不能重复。它用来让框能找到自己对应的图片。
-   **`category['id']`（类别 ID / 主键）**
    -   **作用**：每个类别的**编号**（比如 `1: 'cat'`, `2: 'dog'`）。
    -   为什么不直接在框里写 `"name": "cat"` 字符串？因为存数字 ID 占体积小、检索匹配快。
-   **`annotation['id']`（标注框 ID / 主键）**
    -   **作用**：每个画出来的矩形框自身的**唯一序号**。
    -   就算一张图上有 10 个框，每个框也必须有独一无二的 `id`（在常规目标检测中基本用不到它，但 COCO 规范要求保留作为主键）。
-   **`annotation['image_id']` 与 `annotation['category_id']`（外键 / 关联线）**
    -   **作用**：**连线工具**。
    -   `image_id`: 明确这个框是属于哪张图片的。
    -   `category_id`: 明确这个框圈出的物体到底是什么类别。





COCO 数据集可以使用 `json.load()` 直接读取，读取之后的结果形如：

```json
data = {
    # 1. 图片信息列表（list of dict）
    'images': [
        {
            'id': 101,                  # 图片唯一 ID
            'file_name': '000101.jpg',   # 对应的图片文件名
            'width': 640,               # 宽（像素）
            'height': 480               # 高（像素）
        },
        {
            'id': 102,
            'file_name': '000102.jpg',
            'width': 1280,
            'height': 720
        }
    ],

    # 2. 类别字典列表（list of dict）
    'categories': [
        {
            'id': 1,                    # 类别原始 ID
            'name': 'cat',              # 类别名
            'supercategory': 'animal'   # 大类（通常不用管）
        },
        {
            'id': 2,
            'name': 'dog',
            'supercategory': 'animal'
        }
    ],

    # 3. 标注框列表（list of dict）
    'annotations': [
        {
            'id': 1001,                 # 这个框本身的唯一编号
            'image_id': 101,            # 关联外键：对应 images 里 id 为 101 的图片
            'category_id': 1,           # 关联外键：对应 categories 里 id 为 1 的类别 (cat)
            'bbox': [100, 150, 80, 60], # [x_min, y_min, width, height] 绝对像素坐标
            'area': 4800,               # 框的面积（80 * 60）
            'iscrowd': 0                # 0 表示单目标，1 表示密集重叠群组（目标检测通常为 0）
        },
        {
            'id': 1002,
            'image_id': 101,            # 同样属于 101 这张图片（说明 101 号图里有猫也有狗）
            'category_id': 2,           # 对应 dog
            'bbox': [220, 180, 110, 95],
            'area': 10450,
            'iscrowd': 0
        }
    ]
}
```



同时，由于这种形式很常见，可以使用 `pycocotools` 包来直接读取数据集，详情自查API

### **2. YOLO 格式的转换计算与表示（TXT）**

YOLO 需要两个计算步骤：

-   **算中心点坐标（像素）**：

    xcenter=xmin+2w=100+2200=200

    ycenter=ymin+2h=50+2100=100

-   **归一化为 0~1 的比例（除以整图宽高）**：

    xnorm=Wxcenter=1000200=0.200000

    ynorm=Hycenter=500100=0.200000

    wnorm=Ww=1000200=0.200000

    hnorm=Hh=500100=0.200000

生成对应的 **`cat_dog.txt`** 文件内容（只有 1 行）：

Plaintext

```
0 0.200000 0.200000 0.200000 0.200000
```

>   格式含义：`类别索引(0代表cat) 中心X比例 中心Y比例 宽度比例 高度比例`

**为什么要这么设计？**

-   **COCO 的初衷**：作为学术界通用基准测评集，JSON 结构化强，一个文件能包含分割掩码（Segmentation Polygon）、关键点（Keypoints）、图像版权等丰富属性。
-   **YOLO 的初衷**：为了极致的训练速度。网络在训练过程中会不断做多尺度缩放（如把图片 Resize 成 640×640 或 320×320）。**归一化的相对比例不会随图片缩放而改变**，可以直接乘上缩放后的尺寸，省去了反复重算绝对像素坐标的开销。







这里贴一个如何将 coco 数据集转换为 yolo 格式的数据集的 python 简单实现：

```python
import json
import os
from pathlib import Path

def coco_to_yolo(json_path, output_dir):
    """
    参数说明:
    json_path: 你的 COCO 标注文件路径，例如 'train.json'
    output_dir: 转换生成的 txt 标签文件要保存的文件夹路径
    """
    
    # 1. 打开并读取 COCO 的 JSON 文件
    print(f"正在读取 JSON 文件: {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 如果保存 txt 的文件夹不存在，自动创建出来
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 建立【类别 ID】到【从 0 开始的连续索引】的映射
    # 原因：COCO 的 category_id 可能是 [1, 3, 5] 这种不连续的数字
    # 但 YOLO 要求类别编号必须是从 0 开始的连续整数 (0, 1, 2, ...)
    cat_mapping = {category['id']: index for index, category in enumerate(data['categories'])}
    
    # 4. 把 images 列表变成字典，方便后面根据 image_id 快速查到图片的尺寸和文件名
    # 结构形如: { 101: {'file_name': 'dog.jpg', 'width': 800, 'height': 600}, ... }
    images_dict = {img['id']: img for img in data['images']}

    # 5. 初始化一个空字典，用来按图片归类标注信息
    # 结构形如: { 101: [标注1, 标注2], 102: [标注1], ... }
    img_to_annotations = {img['id']: [] for img in data['images']}
    
    # 遍历 JSON 中的所有标注框，把它们塞进对应 image_id 的列表中
    for ann in data.get('annotations', []):
        img_to_annotations[ann['image_id']].append(ann)

    # 6. 开始逐张图片生成对应的 YOLO txt 标注文件
    for img_id, annotations in img_to_annotations.items():
        # 获取当前图片的元数据（文件名、宽、高）
        img_info = images_dict[img_id]
        img_w = img_info['width']   # 图片原始宽度（像素）
        img_h = img_info['height']  # 图片原始高度（像素）
        
        # 获取去除后缀后的纯文件名（例如 'dog.jpg' -> 'dog'）
        file_stem = Path(img_info['file_name']).stem
        
        # 拼接对应的 txt 文件保存路径（例如 'labels/train/dog.txt'）
        txt_path = os.path.join(output_dir, f"{file_stem}.txt")

        # 写入 txt 文件
        with open(txt_path, 'w', encoding='utf-8') as out_file:
            for ann in annotations:
                # 提取 COCO 原始坐标: [x_min, y_min, w, h]（左上角像素坐标和框的宽高）
                x_min, y_min, box_w, box_h = ann['bbox']
                
                # 转换类别 ID 为 YOLO 需要的从 0 开始的索引
                cls_index = cat_mapping[ann['category_id']]
                
                # ------ 核心算法：COCO 坐标转 YOLO 归一化坐标 ------
                # ① 计算目标框的中心点坐标 (像素)
                x_center_pixel = x_min + (box_w / 2.0)
                y_center_pixel = y_min + (box_h / 2.0)
                
                # ② 归一化：除以整张图的宽高，把数值缩放到 0 ~ 1 之间
                x_center = x_center_pixel / img_w
                y_center = y_center_pixel / img_h
                w_norm = box_w / img_w
                h_norm = box_h / img_h
                # --------------------------------------------------
                
                # 按照 YOLO 格式写入一行: "类别 x_center y_center w h"，保留 6 位小数
                out_file.write(f"{cls_index} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    print(f"✅ 转换完成！txt 文件已保存在: {output_dir}")
    print("👉 你的数据类别按顺序为:", [c['name'] for c in data['categories']])
    print("   (在后面的 yaml 配置文件中，names 列表必须严格按照这个顺序写！)")


# ================== 使用示例 ==================
# 根据你自己的实际路径进行修改：
if __name__ == '__main__':
    # 转换训练集
    coco_to_yolo(
        json_path='my_dataset/annotations/train.json', # 你的 train.json 路径
        output_dir='my_dataset/labels/train'           # 转换后存放 txt 的目录
    )
    
    # 转换验证集
    coco_to_yolo(
        json_path='my_dataset/annotations/val.json',   # 你的 val.json 路径
        output_dir='my_dataset/labels/val'             # 转换后存放 txt 的目录
    )
```





# 不平衡数据的处理方法

如何处理不平衡的数据？

在实际应用中，数据不平衡是一个常见的问题。例如在目标检测任务中，正负样本比例可能达到 1:1000。在医疗领域，疾病诊断任务中，正负样本比例可能达到 1:10000 等等。所以如何处理这一种样本类别不平衡很重要。

处理不平衡数据的方法主要可以分为四大维度：**数据层面（重采样）**、**算法层面（加权与损失函数）**、**集成学习与特定范式**，以及**评估与决策阈值调整**。

**一、 数据层面：重采样技术（Resampling）**

通过增减样本数量使类别比例趋于平衡，通常借助 `imbalanced-learn` 库实现。

-   **过采样（Over-sampling）：增加少数类样本**
    -   **随机过采样（Random Over-sampling）：** 简单复制少数类样本（容易导致过拟合）。
    -   **SMOTE（合成少数类过采样）：** 在少数类样本之间进行特征空间线性插值。
    -   **Borderline-SMOTE / ADASYN：** 聚焦于分类决策边界附近的少数类，或者根据样本合成难度动态分配生成权重。
-   **欠采样（Under-sampling）：减少多数类样本**
    -   **随机欠采样（Random Under-sampling）：** 随机丢弃多数类样本（计算快，但可能损失重要信息）。
    -   **Tomek Links / Edited Nearest Neighbours (ENN)：** 找出边界附近容易混淆的样本对并剔除多数类，起到清洗边界噪声的作用。
-   **混合采样（Combination）：**
    -   结合过采样与欠采样（如 **SMOTE-Tomek** 或 **SMOTE-ENN**），先合成少数类，再清理重叠边界样本。

>   **注意：** 所有重采样操作**只能应用于训练集**，验证集与测试集必须保持原始数据分布，以防数据泄漏。

**二、 算法层面：代价敏感与损失函数调整**

无需修改原始数据量，直接在模型训练过程中加大对少数类误判的惩罚。

-   **类别权重（Class Weights / Cost-Sensitive Learning）：**

    -   给少数类分配更高的损失权重。例如 Scikit-Learn 中分类器的 `class_weight='balanced'`，其权重通常按反比计算：

        $$w_c = \frac{N}{K \cdot N_c}$$

        （$N$ 为总样本数，$K$ 为类别数，$N_c$ 为该类别样本数）。

    -   LightGBM / XGBoost 中对应的参数为 `scale_pos_weight` 或 `is_unbalance`。

-   **修改损失函数（Focal Loss）：**

    -   通过引入调节因子 $(1 - p_t)^\gamma$，降低易分类样本（多数类）的损失贡献，让模型专注于难分类的少数类样本（常用于目标检测和高不平衡深度学习任务）。

**三、 集成学习与范式转换**

-   **不平衡集成算法（Balanced Ensembling）：**
    -   **EasyEnsemble / Balanced Random Forest：** 每次从多数类中抽样一个与少数类等量的子集，分别训练弱分类器，最后集成预测。
-   **转换为异常检测（Anomaly Detection）：**
    -   当正负样本比例极度悬殊（如 1:1000 以上，如信用卡欺诈、工业缺陷检测）时，将少数类视为“异常”，改用无监督/半监督模型（如 **Isolation Forest**、**One-Class SVM**、**Autoencoder**）。

**四、 评估指标与阈值微调**

-   **更换评估指标（放弃 Accuracy）：**
    -   使用 **Precision / Recall / F1-Score** 或 **PR-AUC (Average Precision)**，在极度不均衡场景下 PR 曲线通常比 ROC-AUC 更敏感、更真实。
    -   **Macro-F1** 或 **Balanced Accuracy**（在多分类不均衡时常用）。
-   **阈值微调（Threshold Moving / Post-processing）：**
    -   模型输出的概率预测默认以 $0.5$ 作为切分点。在正样本极少的任务中，可以通过网格搜索或基于验证集的 Precision-Recall 曲线，主动将正类的判定阈值调低（例如降至 $0.2$ 或 $0.3$）以提升召回率。

**方法选型参考**

| **不平衡严重程度**              | **样本规模** | **推荐优先尝试的方法**                                       |
| ------------------------------- | ------------ | ------------------------------------------------------------ |
| **轻度/中度** (如 1:5 ~ 1:50)   | 大/中规模    | 调节模型自带的 `class_weight` / `scale_pos_weight`，调整预测阈值 |
| **中度/重度** (如 1:50 ~ 1:500) | 中/小规模    | **SMOTE / Borderline-SMOTE** + 类别加权，或 **BalancedRandomForest** |
| **极度悬殊** (> 1:1000)         | 任意         | 转换为**单分类/异常检测**（Isolation Forest、Autoencoder）或使用 **Focal Loss** |