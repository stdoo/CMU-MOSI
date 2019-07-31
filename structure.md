# align method
作用：让数据集中的其他模态向 reference 模态对齐。
逻辑：
1. 确保数据集中所有模态拥有相同的 vid
2. 将其他模态以 vid 为基础，组织 intervals 和 features，并保证 intervals 的顺序。
3. 对 reference 模态的每个 interval ，求出其他模态在这个 interval 之内的所有 intervals，及其对应的特征。
4. 由于 intervals 是连续的，所以在截断时只有首尾的 intervals 会受影响，直接用原来的特征就好。
5. Collapse intervals：取所有 intervals 中的最小值和最大值作为 Collapse 后的 interval ，很好理解，因为 intervals 之间是连续的，即前一个 interval 的结束时间为下一个 interval 的开始时间。
6. Collapse features：支持多个 Collapse 函数，然后把这些函数得到的结果串联起来。
7. 最后，所有模态的 vid 都会加上 interval 编号，如 vid[0] 。

结果：变成了每个 word 对应的 intervals 和 features 。

