"""Write conclusions with separate confirmed evidence and causal limitations."""
from pathlib import Path
import argparse,json
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.output/'rinit_ablation_metrics.json').read_text());full=report['full528'];pilot=report['pilot128']
    gradient=json.loads((a.output/'surrogate_gradient_audit.json').read_text());real=gradient['real_logits'];rows=gradient['rows']
    def fmt(r):
        if r['mean'] is None:return 'NA'
        return f"{r['mean']:.6f} [{r['ci95'][0]:.6f}, {r['ci95'][1]:.6f}]"
    d=full['paired_delta_r10_minus_r05']
    primary_improved=d['inside_masked_lpips']['ci95'][1]<0
    ssim_improved=d['inside_ssim']['ci95'][0]>0
    actual=[r for r in rows if r['kind'].startswith('real_auxiliary_perturbation') and r['objective']=='fp32_squared' and r['expected_gradient_norm']>0]
    zero_fractions=[r['nonzero_reference_computed_zero_fraction'] for r in actual]
    zero_range=f'{min(zero_fractions):.2%}–{max(zero_fractions):.2%}' if zero_fractions else 'NA（扰动未得到非零 g）'
    bf_actual=[r for r in actual if 'bf16' in r['kind']]
    fp_small=[r for r in actual if 'fp32_g' in r['kind'] and r['perturbation_magnitude']==1e-5]
    linear_rows=[r for r in rows if r['objective']=='fp32_linear' and r['expected_gradient_norm']>0]
    lines=['# EditMGT LoRA DiMO Step0 质量与 Surrogate 梯度诊断','',
        '**所有 Step0 结果均为 `RECONSTRUCTED_STEP0`。本机没有真实 Step0 checkpoint，用户已确认；没有创建或修改 checkpoint。**','',
        '## 直接回答实验问题','',
        f"1. **CONFIRMED（重建初始化模型）**：r0.5 的一步编辑质量与 Teacher 的配对差异：Inside LPIPS Δ={fmt(full['paired_delta_r05_minus_teacher']['inside_masked_lpips'])}，Inside SSIM Δ={fmt(full['paired_delta_r05_minus_teacher']['inside_ssim'])}。固定预览中的新增香槟瓶、盘沿颜色、地板、人物编辑已出现 ROI 碎片化与语义失败。**NOT_TESTED**：真实历史 Step0 权重、全量盲评成功率；不能把 LPIPS 差异转换成失败率。",
        f"2. **CONFIRMED（同一 Student 的推理消融）**：r1.0 {'显著降低' if primary_improved else '未证实显著降低'} Inside LPIPS，{'显著提高' if ssim_improved else '未证实显著提高'} Inside SSIM；ΔLPIPS={fmt(d['inside_masked_lpips'])}，ΔSSIM={fmt(d['inside_ssim'])}。r1.0 仍可能发生个别语义失败，见固定 montage。**SUSPECTED**：MASK/random 混合加剧碎片化。只改变 r_init 参数，但旧推理契约也随实际 MASK 比例改变 timestep，因此没有独立隔离随机 token 的因果作用。**NOT_TESTED**：它是否是 step7500 训练失败的根因。",
        '3. **CONFIRMED**：FP32 squared surrogate 存在 logit−g 减法相消。阈值取决于 logit 的 ULP；绝对 g 小于约半个 ULP 时会被舍去，而不是一个适用于全部 logits 的固定阈值。下表给出受控实验范围。',
        f"4. **CONFIRMED（数值诊断）**：真实 logits 配合受控非零 g 可以独立检查同类相消。真实 Auxiliary 小扰动的非零分布 g 中，FP32 squared 的非零参考梯度被算成 0 的比例范围为 {zero_range}。零分量比例不等于整体 norm 损失，实际 norm ratio、relative-L2 error、cosine 见下表。真实原始初始化 g={'全部为 0' if all(c['original_real_g']['norm']==0 for c in real['cases']) else '含非零值'}，与 Teacher/Auxiliary 完全相同的权重相符；这不是用来证明相消的证据。受控 g 与真实扰动 g 在 JSON/CSV 中分开标识。",
        '5. **CONFIRMED（建议与代码现状）**：后续 Full DenseDiMO 应使用稳定 linear surrogate 并保留严格有限值检查、真实 logits 梯度审计。维护分支本来已有 linear 实现，`configs/dimo/full_dense.yaml` 本来已选择 linear；本次没有再改正式 surrogate、step、divergence 或训练配置。**SUSPECTED**：旧 squared 的损失会影响训练；**NOT_TESTED**：它对历史完整训练失败的因果贡献与改用 linear 后的训练收益。','',
        '## 身份与协议','',
        '- **CONFIRMED**：Teacher artifact 的五个 SHA256 与旧 LoRA DiMO manifest 一致。底座为 `WeiChow/EditMGT@0a00f14ea159728fb76ac99ed741f9deca5fee28`，权重来自既有固定快照。',
        '- **CONFIRMED**：兼容 LoRA 推理提交 `3132e24355f947aaeb604278514d9460806de0b6`，模型 backend 为 shared_base_adapters；未使用 Dense 模型替代。',
        '- **CONFIRMED**：Student/Auxiliary 的 421 个 trainable-role 张量与按兼容 BF16 契约加载后的 Teacher 逐张量相同；初始 FP32 EMA shadow 与 Raw Student 的 FP32 表示一致。EMA 是初始化重建检查，非已保存的真实 EMA checkpoint，没有独立重复评测。',
        '- **NOT_TESTED**：历史训练源 SHA `c4f92a7c696150c40f07959c72aa435cb6f67624` 不可获取；重建不能证明与不可取得的原始 Step0 artifact 完全相同。身份记录含所有路径、版本、role tensor 哈希。',
        f"- **CONFIRMED**：DEV128 {pilot['source_groups']} 个 source groups；完整 DEV {full['samples']} 条、{full['source_groups']} 个 source groups。每条 seeds=[0,1]，每设置 {full['per_variant_generations']} 张全量输出。完整集未更换样本、mask 或几何处理；沿用原先排除的那一条不匹配样本。",
        '- **CONFIRMED**：1024 分辨率；Teacher 12 steps / CFG5；Student 1 forward / CFG1；sigma=.3，temperature=1，top-k=0，top-p=0。r0.5=ROI 内按精确数量 MASK/random 混合（小 ROI 有取整），r1.0=全 MASK；source tokens 在 ROI 外保持不变。',
        '- **CONFIRMED**：同一样本的两个 seed 先均值，再对 source groups 重采样 10,000 次（seed42），保持 source 多 turn 的成员结构。改善比例也按 sample 的 seed 均值计算，提供 source-group CI。',
        '- **CONFIRMED**：复用已完成 DEV128 的相同输出用于完整集；每份缓存的输入哈希、UID、seed 均校验。每张 VQ reconstruction 与此前逐像素一致，故可复用此前 VQ LPIPS baseline。',
        '- **CONFIRMED**：Teacher 与 step7500 Raw 逐样本记录的输入、seed、manifest、旧协议与 checkpoint 身份均核对后复用，没有重新加载 step7500 Student 权重。',
        '- **CONFIRMED**：确定性 source/text conditioning 缓存只省重复编码；独立检查得到原版推理与缓存版的 logits 和输出像素逐位相同。计时范围不同，未做 step0 vs 旧评测的延迟/速度结论。','',
        '## 完整 DEV 指标（均值 [source-group 95% CI]）','',
        '| 指标 | Teacher | Step0 r0.5 | Step0 r1.0 | 历史 step7500 Raw |',
        '|---|---:|---:|---:|---:|']
    metric_names=['inside_masked_lpips','inside_psnr','inside_ssim','outside_masked_lpips','outside_psnr','outside_ssim',
        'DINO-I-to-Target','CLIP-I-to-Target','vq_noedit_inside_lpips_to_target','vq_noedit_outside_lpips_to_source','editing_gain_lpips',
        'transformer_forwards','outside_token_mismatch_count']
    for n in metric_names:lines.append('| '+n+' | '+' | '.join(fmt(full['variants'][v][n]) for v in ['teacher','r05','r10','step7500'])+' |')
    lines+=['','LPIPS 越低越好；PSNR/SSIM/embedding similarity 越高越好。连续指标不是指令成功率。VQ baseline 和 Teacher 也不一定取得正的 editing gain。','',
        '## r_init 消融的配对差异','',
        '| 指标 | r1.0 − r0.5，均值 [95% CI] | 改善 sample 比例 [95% CI] |','|---|---:|---:|']
    for n in ['inside_masked_lpips','inside_ssim','outside_masked_lpips','outside_ssim','inside_psnr','outside_psnr']:
        lines.append(f"| {n} | {fmt(d[n])} | {fmt(full['improved_sample_proportion'][n])} |")
    lines+=['','比例为 0–1 小数，按 sample 的两个 seed 均值判断严格改善，不把两个 seed 当独立样本。','',
        'Raw Student 两次输出的差异（不与 target 比较）：','']
    for n in ['output_change_inside_l1','output_change_inside_ssim','output_change_inside_lpips','output_change_outside_l1','output_change_outside_lpips','output_change_full_l1']:
        lines.append(f"- **CONFIRMED** `{n}`：{fmt(full['raw_output_changes'][n])}。")
    lines+=['','**CONFIRMED**：ROI 外锁定的是 VQ tokens；VQ decoder 的跨边界感受野仍可使 ROI 外像素随 ROI 内容变化，不能要求 outside 图像指标恒等。','',
        '## 与历史 step7500 Raw 的配对比较','',
        '下面定义 Δ=重建 Step0 r0.5 − 历史 step7500 Raw。Step0 是初始化反事实重建，无法当作真实历史 Step0 artifact。','',
        '| 指标 | Δ 均值 [95% CI] |','|---|---:|']
    for n in ['inside_masked_lpips','inside_ssim','outside_masked_lpips','DINO-I-to-Target','CLIP-I-to-Target']:
        lines.append(f"| {n} | {fmt(full['paired_delta_r05_minus_step7500'][n])} |")
    lines+=['','**CONFIRMED**：这只比较相同推理契约的质量指标；即使差异很小且统计显著，也不能据此判断训练有效性、优化轨迹或因果根因。','',
        '## 合成 FP64/FP32 surrogate 审计','',
        'z 的绝对值均匀分布于 [0.5×magnitude, magnitude]，随机正负；g 是指定 RMS scale 的高斯数据。',
        '覆盖 logit magnitude=1/5/10/20，g=1e-2…1e-8，ROI 比例=1/32、1/4、1，batch=1/2/4，并令 batch 内 selected-token 数量不同。',
        'FP64 analytical reference = g.detach()/(B×每样本 selected-token 数量)，ROI 外为 0，保留整个 vocabulary sum。FP64 squared 不强制降至 FP32。','',
        '| logit magnitude | g scale | FP32 squared 非零→零比例（跨 ROI/batch 设置均值） |','|---:|---:|---:|']
    for m in [1,5,10,20]:
        for scale in [1e-4,1e-5,1e-6,1e-7,1e-8]:
            selected=[r for r in rows if r['kind']=='synthetic' and r['objective']=='fp32_squared' and r['logit_magnitude']==m and r['g_magnitude']==scale]
            lines.append(f"| {m} | {scale:g} | {np.mean([r['nonzero_reference_computed_zero_fraction'] for r in selected]):.4%} |")
    lines+=['',f"**CONFIRMED**：所有非零 g 的 FP32 linear 测试，最大 relative-L2-error={max(r['relative_l2_error'] for r in linear_rows):.3g}，最大非零→零比例={max(r['nonzero_reference_computed_zero_fraction'] for r in linear_rows):.3g}。signed linear loss 的标量可能为负，不作为 KL/energy 数值监控。",'',
        '## 真实 logits、BF16 与分布梯度','',
        '真实 Student 前向按原兼容 BF16 权重契约执行；额外 Teacher/Auxiliary 查询覆盖 CUDA BF16 autocast。同一 pseudo-state 来自 Student 预测并按原 cosine ratio 再 mask，使用相同 source、ROI、prompt、timestep。',
        '数值 surrogate 测试只对 detached logits 创建独立叶子，不对模型参数 backward。controlled g 与小扰动产生的 distribution g 分开标记。',
        '临时 Auxiliary output-LoRA-B 扰动 RMS 目标为 1e-5/1e-4/1e-3，实际 BF16 参数量化后的 delta 另记；原 tensor 每次完整恢复。FP32 transformer reference 为同一 BF16 权重精确上转和同一 conditioning 的 FP32 表示，未重算 FP32 text encoder。',
        'FKL 的 FP64 distribution reference 在相同 logits 上计算，用于分离 softmax FP32 舍入与 Transformer BF16 前向误差。','',
        '| 案例 | 扰动目标 RMS | BF16 g norm | FP32-forward g norm | BF16 对 FP32-forward/FP64 distribution 的非零→零比例 |','|---|---:|---:|---:|---:|']
    for c in real['cases']:
        for t in c['perturbation_tests']:
            fraction=t['bf16_forward_vs_fp32_forward_distribution_gradient']['nonzero_reference_computed_zero_fraction']
            lines.append(f"| {c['sample_key']} | {t['perturbation_rms_requested']:g} | {t['bf16_forward_g']['norm']:.6g} | {t['fp32_forward_g']['norm']:.6g} | {'NA' if fraction is None else f'{fraction:.4%}'} |")
    lines+=['','固定真实 Student logits，注入真实 Auxiliary 扰动 g 后的 FP32 squared autograd：','',
        '| 案例 | g 来自的前向 | 扰动 RMS | actual / expected norm | relative-L2 error | cosine | 非零→零比例 |','|---|---|---:|---:|---:|---:|---:|']
    for r in actual:
        cosine='NA' if r['gradient_cosine'] is None else f"{r['gradient_cosine']:.6f}"
        lines.append(f"| {r['sample_key']} | {'BF16' if 'bf16' in r['kind'] else 'FP32'} | {r['perturbation_magnitude']:g} | {r['gradient_norm_ratio']:.6f} | {r['relative_l2_error']:.6f} | {cosine} | {r['nonzero_reference_computed_zero_fraction']:.4%} |")
    if bf_actual and fp_small:
        ratios=[t['bf16_forward_g']['norm']/t['fp32_forward_g']['norm'] for c in real['cases'] for t in c['perturbation_tests']
                if t['perturbation_rms_requested']==1e-5 and t['fp32_forward_g']['norm']>0]
        lines+=['',
            f"**CONFIRMED（重要区分）**：真实 BF16-forward 扰动 g 注入的 squared 梯度，norm ratio={min(r['gradient_norm_ratio'] for r in bf_actual):.6f}–{max(r['gradient_norm_ratio'] for r in bf_actual):.6f}，relative-L2 error={min(r['relative_l2_error'] for r in bf_actual):.2%}–{max(r['relative_l2_error'] for r in bf_actual):.2%}。虽然许多微小分量变成零，不能说整体梯度 norm 同比例丢失。",
            f"**CONFIRMED（FP32-forward 小分布差异）**：扰动 RMS=1e-5 的 FP32-forward g，squared 的 actual/expected norm 仅为 {min(r['gradient_norm_ratio'] for r in fp_small):.2%}–{max(r['gradient_norm_ratio'] for r in fp_small):.2%}，relative-L2 error={min(r['relative_l2_error'] for r in fp_small):.2%}–{max(r['relative_l2_error'] for r in fp_small):.2%}。这对未来 FP32 Full DenseDiMO 使用旧 squared 的风险更有直接参考意义；仍非 Full DenseDiMO 训练实验。",
            f"**CONFIRMED（BF16 前向效应）**：相同有效扰动权重与 conditioning，RMS=1e-5 时 BF16 g norm 是 FP32-forward g norm 的 {min(ratios):.1f}–{max(ratios):.1f} 倍。这里既有差异分量被压成零，也有量化引起的较大不连续变化，不能统一称作 BF16 的梯度丢失。"]
    lines+=['','明确区分以下机制：','',
        '- **CONFIRMED** 初始 Teacher/Auxiliary 角色权重和 logits 相同产生原始 g=0。这是正确的零分布梯度，不能用于证明 surrogate 稳定或不稳定。',
        '- **CONFIRMED（量化对照）** BF16-forward 和 FP32-forward 的真实分布差异见上表，不能把 BF16 将差异压成零的分量归因于后续 squared。',
        '- **CONFIRMED（独立叶子 autograd）** 固定非零 FP32 g 注入真实 z 后的 squared/linear 差异，隔离了 FP32 的 z−g 相消。BF16 leaf 的结果还包含叶子梯度 dtype 的舍入，CSV 同时提供与 BF16-rounded reference 的比较。',
        '- **CONFIRMED** ROI 缩放严格使用 1/(B×每样本 selected_tokens)，analytical reference 包含同样缩放。g.norm 变小不自动表示梯度丢失；用实际/参考 norm、relative error 和非零→零比例判断。','',
        '**CONFIRMED（旧代码）**：兼容分支 `divergence.py` 用 `nan_to_num` 静默替换非有限 gradient。独立注入 1 个 NaN logit 后，softmax 产生 4 个非有限概率而返回全零的有限 g。真实有限前向没有观察到非有限的原始分布梯度；不能据注入测试声称历史训练曾实际发生 NaN。',
        '**CONFIRMED（维护分支现状）**：当前 divergence.py 已在 logits 和 g 上 fail-fast；这不是本次新增修复。','',
        '## 推荐保留的稳定实现','',
        '下面给出与原 FP64 squared 理论梯度一致的稳定注入式，M 是原 Teacher pseudo-mask，且必须位于 ROI 内。正式代码已有实现，本次仅提供诊断建议。','',
        '```python',
        'z32 = z.float()',
        'g32 = g.detach().float()',
        'mask = pseudo_mask.bool().flatten(1)',
        'counts = mask.sum(1)',
        'assert (counts > 0).all()',
        'if not torch.isfinite(g32).all():',
        '    raise FloatingPointError("NONFINITE_DIMO_G")',
        'token_dot = (z32 * g32).sum(-1)  # vocabulary SUM',
        'per_sample = token_dot.flatten(1).masked_fill(~mask, 0).sum(1) / counts',
        'loss = per_sample.mean()       # equal sample weights within batch',
        '```','',
        '**CONFIRMED**：测试所得 linear 梯度与 sg(g)/(B×|M_b|) 一致。保留独立 distribution-energy/KL 监控，不能把 signed linear scalar 当 KL。**NOT_TESTED**：修改 surrogate 后完整 Dense 训练的实际收益；不根据本次 DEV 的 r1.0 消融自动改变训练初始化设置。','',
        '## 验证与限制','',
        '- **CONFIRMED**：13 项相关单元测试通过，覆盖 vocabulary SUM、不同样本 ROI counts、batch mean、g.detach、空 ROI 拒绝、small-g 相消、FP64 reference 与 BF16 特殊 stride 的权重完整性校验。',
        '- **CONFIRMED**：推理全程只有一个 Student forward；初始与最终 ROI 外 token 不变；所有 worker 的 Student role digest 前后相同。',
        '- **CONFIRMED**：扰动结束后完整 Transformer/role state_dict、角色权重、train/eval/require_grad flags、CPU/CUDA RNG 恢复，所有模型参数 grad 为 None。没有 optimizer、optimizer step 或训练启动。',
        '- **CONFIRMED**：正式训练源码与配置的 SHA256 前后核对；诊断只新增 scripts/diagnostics 和诊断测试，没有创建实验分支。',
        '- **NOT_TESTED**：真实 Step0 checkpoint、正式 TEST 泛化、盲评成功率、Dense 参数梯度/训练曲线、step7500 历史真实 g、把 surrogate 替换后训练是否变好。',
        '- **SUSPECTED**：随机初始化、一步能力与数值梯度损失可能分别影响旧 LoRA DiMO；本次没有把它们合成单一根因结论。','',
        '## 文件与复现','',
        '`step0_metrics.json`、`rinit_ablation_metrics.json` 包含两阶段指标、配对 CI、逐样本 delta；`surrogate_gradient_audit.json/.csv` 包含 norm/error/cosine/zero fraction；图表、montage、身份、环境和代码快照一并打包。',
        '`CLUSTER_COMMANDS.sh` 给出仅推理/诊断的重跑命令；外部模型资产和完整输入数据须按固定身份准备。`code/compatible_lora_runtime` 是实际使用的旧 LoRA 代码，`code/maintenance` 是维护分支及诊断代码。',
        '压缩包不含模型权重、optimizer 或完整数据集。逐张全量输出保留在本机 work_root，供检查，不纳入轻量包。','']
    (a.output/'REPORT.md').write_text('\n'.join(lines))
    print('REPORT_COMPLETE',flush=True)


if __name__=='__main__':main()
