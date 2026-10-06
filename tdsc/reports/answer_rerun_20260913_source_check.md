# A 组完整答案重跑：源数据核对

核对日期：2026-09-13。范围是 `manifests/conference.json`、`data/conference`、父项目 `captcha_data`、会议 loader/scorer，以及相关原始标注。没有读取旧模型输出，没有调用模型，没有修改输入、GT、评分阈值或样本集合。用户澄清后，交互界面适配设计已停止；本报告仅用于完整答案输出的重跑。

**结论：未发现 TDSC 数据复制或 manifest 尺寸错误；发现 1 道在原图内作答约束下无可接受解的源标注。该问题不是 JSON 格式错误，也不是模型定位能力的负面证据。**

## 1. 输入与标签副本完整

A 保留 141 道题：Dice_Count 11、Click_Order 10、Place_Dot 32、Pick_Area 30、Patch_Select 10、Rotation_Match 48。

- Manifest 共引用图片 199 次，涉及 116 个唯一文件。所有文件均可解码，实际尺寸全部与 manifest 一致。
- 这 116 个输入文件逐一与父项目 `captcha_data` 中的对应文件比较，字节完全一致。
- 六类 `ground_truth.json` 副本也均与父项目对应文件字节完全一致。
- Click_Order 输入顺序是 `[reference, main]`；其答案坐标属于 main 图。Rotation_Match 的静态输入顺序是 `[reference, object_base]`。现有或新下载的候选角度图不属于此次完整答案重跑的静态输入。

证据入口：`src/tdsc_eval/tasks.py:32`、`src/tdsc_eval/_conference/run_eval.py:2285`、`src/tdsc_eval/_conference/run_eval.py:2419`。

## 2. 源标注冲突：Click_Order/image8.png

记录 ID：`conference/Click_Order/image8.png`。Manifest 条目从 `manifests/conference.json:1086` 开始；父项目和副本标注条目均从 `Click_Order/ground_truth.json:170` 开始。

原主图尺寸为 **600 × 390**。原标注为：

```json
{
  "answer": [[468,118], [280,465], [132,121], [470,380]],
  "order_image": "order7.png",
  "tolerance": 50
}
```

第二个目标点 `(280,465)` 位于原图下方。按整数像素范围 `0 ≤ x ≤ 599, 0 ≤ y ≤ 389`，任意图内点距它至少 76 像素，超过该题 50 像素容差。即使将连续图像边界宽松解释为 `y ≤ 390`，最小距离也为 75，结论不变。因此，不存在同时满足原提示的图内坐标要求和原 Click_Order 判分的完整答案。

这不等于评分函数无法计算。`evaluate_pass1` 在 `src/tdsc_eval/_conference/run_eval.py:2569` 仅检查点数、顺序和欧氏距离，没有额外检查主图边界；显式返回越界 GT 数值仍可被它判为正确。准确分类应为：**源标注与图内作答约束冲突，无法作为正常可完成的图内定位题解释**，而不是解析失败或普通答错。

已检查的原始来源：

1. 父项目 `captcha_data/Click_Order/ground_truth.json` 与 `image8.png`；
2. TDSC 副本 `data/conference/Click_Order/ground_truth.json` 与 `image8.png`；
3. `expanded_captcha_data/phase04_2/adaptive_evaluator_slice/Click_Order` 中的对应旧数据副本。

三处条目相同，三处图片均为 600 × 390；父项目和 TDSC 的标注文件、image8 文件分别字节相同。原 image8 文件没有嵌入尺寸、旋转或坐标变换元信息。标注未提供原始画布尺寸、缩放比例或行列坐标约定；loader 在 `run_eval.py:2312` 明确将列表第一项读为 x、第二项读为 y。

没有证据支持自动修正。整体交换 x/y 会把第一点变成 `(118,468)`、第四点变成 `(380,470)`，仍越界；现有文件也没有支持按某个比例重缩放的来源记录。本报告不根据图中物体位置猜测替代坐标。

保留处理：不修改该条 GT，不扩大容差，不拉伸图片，不删除题目或改变 141 题的原分母。重跑记录可另附源有效性标记；解释该题结果时应单列上述冲突，避免把它合并为模型格式或定位错误。若未来修订标注，应另有来源证据和独立版本，不能静默覆盖本次原协议结果。

## 3. 不应误当作格式修复的问题

以下是现有任务和评分契约，需要原样保留；它们不是本次发现的新增工程故障。

| 类型 | 原输入/完整答案语义 | 原判分及边界 |
|---|---|---|
| Dice_Count | 统计全部骰子顶面数字或可见点数，输出一个整数；不是把侧面也累加 | 整数与 `sum` 精确相等，`run_eval.py:2483` |
| Click_Order | 两张独立图；按参考顺序输出主图中的完整点列表 | 点数及顺序必须一致；image8 容差 50，其余 9 题为 40；不能替换成 B 的矩形命中规则 |
| Place_Dot | 在与汽车起点相连路径的终点给一个点 | 欧氏距离圆形容差，25 像素有 2 题、20 有 29 题、15 有 1 题；`run_eval.py:2511`，不能统一成一个值或方框 |
| Pick_Area | 对最大虚线区域给一个点 | 使用原 `area_box` 的闭区间矩形命中，`run_eval.py:235`、`:2563`；并非重建真实曲线多边形后评分 |
| Patch_Select | 所有 10 题均为 5 × 5；完整答案是 0-based、row-major 的所选索引集合 | 索引范围 0–24，按完整集合精确比较，`run_eval.py:2506`；不能套 B 的 4 × 4 |
| Rotation_Match | 两张原图，输出一个角度；不浏览候选或新增控件 | 环形角度误差不超过逐题 5°；没有另外接受 180° 对称答案的规则，`run_eval.py:2590` |

Pick_Area 的 image2、image3、image8 标注边缘写到 500，而图的最后整数像素是 499；image5 的区域为 `[[200,300],[510,500]]`。这些矩形都与原图有非空且足够大的交集，存在图内可接受答案，故不能与 Click_Order/image8 的无图内可接受解混为一类。应保留现有闭区间矩形评分，不裁剪或重新标注。

除 Click_Order/image8 外，本次对全部 A Click_Order 和 Place_Dot GT 点的几何核对未发现越界目标；所有 Pick_Area 标注均与原图相交。此项检查不证明全部视觉标注正确，只排除了这些可直接计算的几何冲突。

普通输出格式问题应在答案解析层独立记录，例如 JSON 包装、键名或坐标表示；源标签不可达属于数据有效性问题，不能用补键、数值归一化、坐标交换或放宽判分来掩盖。本次未读取模型回答，因此没有在此报告新增或重算任何模型格式失败统计。

## 4. A Rotation 的角度正方向

原始素材与标签一致支持**从当前对象图出发的顺时针附加旋转角度**：父项目 `captcha_data/Rotation_Match/cat.png` 为正立图，已有 `cat_90.png` 对应其顺时针 90° 旋转；该候选与原图顺时针 90° 转置的 decoded RGB 逐像素一致。`direction_1.png` 的手指向右，原标签为 90°；`direction_3.png` 向左，标签为 270°；`direction_7.png` 向上，标签为 0°。六个对象的原标签均使用相同参考方向映射。这里依据的是原始素材和原标注的一致性，**没有找到或声称找到这些素材的原始生成代码**，也没有依据模型正确率选择方向。

已核对 `src/tdsc_eval/answer_eval.py:20` 的提示修改：A 的全部 48 道 Rotation 现在明确 image 1 为参考、image 2 为当前对象，并要求 additional CLOCKWISE rotation、单位 degrees、范围 `[0,360)`；图像顺序与原 loader 一致。原提示中的 180° 对称性及选择最小正确角度示例已经移除，因为原 scorer 仅支持一个 GT 角度的 5° 环形容差，没有额外对称性接受集合。这两处澄清与源任务约定一致，未修改图像、GT 或 scorer；B 原有的顺时针提示保持不变。
