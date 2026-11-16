跨区域智慧教育应用成效与公平评估平台

scale_reviews.open 使用一张已发布评分卡及固定评分门槛建立学校扩容评议，评分不足的学校不进入待审队列。
每轮评议固定比较人口、证据指纹和总名额；decide 确认或暂缓单校申请，安全门禁仍为 hold/reject 时不能批准。
多校共享本轮名额，完成结论不可覆盖。scale_reviews.inspect 可重开数据库核对候选、名额和历史依据。

同批评分卡：run 将同一次比较中的学校决策、记录版本、证据指纹
和评分固化到 cohort_card；在同一数据库内可通过 repository.cohort_card(run_id)
重读已发布评分卡。清单重放幂等，新证据不得静默覆盖旧卡。

缺失模式诊断与敏感性比较：run 在评分卡之外导出 evidence-review.json，
分层统计缺成绩/缺使用记录与资格、同意范围、地区差异的关联，并在
complete_case、published_zero_fill、consent_restricted、stratum_mean、
best_case、worst_case 六种处理假设下并列比较成效与公平性估计；
报告说明结论由哪些观测支持、依赖哪些假设，不自动采用最有利的处理方法，
已发布评分卡截止口径对应 published_zero_fill。

扩容证据审查：scale_reviews.review_evidence 依据当前证据库为评议附加证据审查，
判断可支持的学校范围；无法比较的群体不能直接判为公平，同意范围外或记录缺失的
学校不能批准。资格或同意更正后调用 re_review 重审待批申请：已发布评分卡的
截止口径（评分、门槛、名额）保持不变，报告把补报带来的估计变化、适用范围变化
与扩容资格变化分开列出，待批学校仍需 decide 确认。缺失场景样例见
fixtures/missing_pattern.json。

运行验收：
python3 -m learning_evidence run --fixture fixtures/acceptance.json --database /tmp/learning_evidence.sqlite3 --output /tmp/learning_evidence-output

核验：
python3 -m learning_evidence verify --database /tmp/learning_evidence.sqlite3 --output /tmp/learning_evidence-output

全量测试：
python3 -m unittest discover -s tests -v

按模块测试：
python3 -m unittest tests.programs.test_contract -v

构建检查：
python3 -m compileall -q learning_evidence tests
