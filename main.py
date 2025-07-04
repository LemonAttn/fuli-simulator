import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CompoundInterest:
    def __init__(self, principal=0.0, annual_rate=0.05, compounding_per_year=365):
        self.principal = principal
        self.annual_rate = annual_rate
        self.n = compounding_per_year

    def growth(self, years, schedule=None, contribution=0.0, contribution_periods=None,
               inflation_rate=0.03,
               management_fee_rate=0.003,
               redemption_fee_rate=0.005,
               schedule_fee_schedule=None,
               model='normal',           # 选择波动模型：'normal'或'jump_diffusion'
               vol=0.15,                # 年化波动率，正态波动模型和跳跃扩散基础波动率
               jump_lambda=0.05,        # 跳跃强度，跳跃事件平均频率（每期概率）
               jump_mu=0.0,             # 跳跃幅度对数正态均值
               jump_sigma=0.02          # 跳跃幅度对数正态标准差
               ):
        fee_per_period = management_fee_rate / self.n
        if schedule_fee_schedule is None:
            schedule_fee_schedule = [
                (30/365, 0.015),
                (180/365, 0.01),
                (1,      0.005),
                (3,      0.003),
                (np.inf, 0.0),
            ]

        if schedule == 'daily':
            contribution_periods = self.n
        elif schedule == 'monthly':
            contribution_periods = 12
        elif schedule == 'yearly':
            contribution_periods = 1

        total_periods = int(years * self.n)
        times = np.arange(total_periods + 1) / self.n
        balances = np.zeros(total_periods + 1)
        principals = np.zeros(total_periods + 1)
        purchase_times = []

        balances[0] = self.principal
        principals[0] = self.principal

        interval = self.n // contribution_periods if contribution_periods else None

        # 新增：记录每年年末的账户余额，用于计算年度收益率
        year_end_balances = []

        # 新增：记录跳跃事件
        jump_count = 0
        jump_magnitudes = []

        peak = balances[0]  # 最大历史余额，用于计算最大回撤
        drawdowns = []  # 每期最大回撤列表

        for t in range(1, total_periods + 1):
            # 扣管理费
            balances[t-1] *= (1 - fee_per_period)

            # 计算每期基础收益率参数
            mu_per_period = self.annual_rate / self.n
            sigma_per_period = vol / np.sqrt(self.n)

            if model == 'normal':
                # 正态分布模型随机收益率
                r_t = np.random.normal(mu_per_period, sigma_per_period)
                jump_occurs = False
            elif model == 'jump_diffusion':
                # 跳跃扩散模型
                r_continuous = np.random.normal(mu_per_period, sigma_per_period)
                jump_occurs = np.random.rand() < jump_lambda
                if jump_occurs:
                    # jump_direction = 1 if np.random.rand() < 0.5 else -1
                    # jump_size = jump_direction * (np.random.lognormal(jump_mu, jump_sigma) - 1)
                    jump_size = np.random.lognormal(jump_mu, jump_sigma) - 1
                else:
                    jump_size = 0.0

                r_t = r_continuous + jump_size
            else:
                raise ValueError("model参数只能是'normal'或'jump_diffusion'")

            # 新增：跳跃统计
            if model == 'jump_diffusion' and jump_occurs:
                jump_count += 1
                jump_magnitudes.append(jump_size)

            # 减去通胀率对应每期的通胀影响
            inflation_per_period = inflation_rate / self.n
            r_real = r_t - inflation_per_period

            # 余额更新
            balances[t] = balances[t-1] * (1 + r_real)
            principals[t] = principals[t-1]

            # 定投扣申购费并记录时间戳
            if interval and (t % interval == 0):
                holding_years = 0
                fee_rate = 0.0
                for max_years, rate in schedule_fee_schedule:
                    if holding_years <= max_years:
                        fee_rate = rate
                        break
                net_amt = contribution * (1 - fee_rate)
                balances[t] += net_amt
                principals[t] += net_amt
                purchase_times.append((t, net_amt))

            # 新增：记录最大回撤
            peak = max(peak, balances[t])
            if peak > 0:
                dd = (balances[t] - peak) / peak
            else:
                dd = 0.0
            drawdowns.append(dd)

            # 新增：记录每年年末余额（用于计算年度收益率）
            if t % self.n == 0:
                year_end_balances.append(balances[t])

        # 扣赎回费
        balances[-1] *= (1 - redemption_fee_rate)
        return times, balances, principals, year_end_balances, drawdowns, jump_count, jump_magnitudes

    def summary(self, years, schedule=None, contribution=0.0, final_balance=None):
        if schedule == 'daily':
            contribution_periods = self.n
        elif schedule == 'monthly':
            contribution_periods = 12
        elif schedule == 'yearly':
            contribution_periods = 1
        else:
            contribution_periods = 0

        total_principal = self.principal + contribution * contribution_periods * years
        interest = final_balance - total_principal
        daily_interest = final_balance * (self.annual_rate / self.n)
        return total_principal, interest, daily_interest

    def year_when_profit_exceeds_principal(self, times, balances, principals):
        profits = balances - principals
        for t, (profit, principal) in enumerate(zip(profits, principals)):
            if profit > principal:
                return times[t]
        return None


if __name__ == '__main__':
    # ============ 参数配置区 ============
    principal = 0                     # 初始本金
    annual_rate = 0.045               # 年化利率（4.5% 适中收益）
    compounding_per_year = 365        # 每年复利次数（日频）
    years = 40                        # 投资年限
    schedule = 'monthly'              # 定投频率
    contribution = 5000               # 每次定投金额
    inflation_rate = 0.0014           # 通胀率（0.14%）
    management_fee_rate = 0.003       # 年管理费率（0.3%）
    redemption_fee_rate = 0.005       # 赎回费率（0.5%）
    risk_free_rate = 0.02             # 无风险收益率（用于计算夏普比率）
    # 模型相关参数（中低风险推荐配置）
    model = 'jump_diffusion'          # 波动模型选择：'normal' 或 'jump_diffusion'
    vol = 0.07                        # 年化波动率 7%，较低波动
    jump_lambda = 0.001               # 跳跃事件概率 0.1% 每期（极端事件罕见）
    jump_mu = 0.0                     # 跳跃幅度对数正态均值，均值0无偏
    jump_sigma = 0.03                 # 跳跃幅度对数正态标准差 3%，小幅跳跃
    # ============ 参数配置区 ============

    ci = CompoundInterest(principal, annual_rate, compounding_per_year)
    times, balances, principals, year_end_balances, drawdowns, jump_count, jump_magnitudes = ci.growth(
        years, schedule, contribution, None,
        inflation_rate,
        management_fee_rate,
        redemption_fee_rate,
        None,
        model,
        vol,
        jump_lambda,
        jump_mu,
        jump_sigma
    )

    total_principal, interest, daily_interest = ci.summary(
        years, schedule, contribution, balances[-1]
    )
    final_amount = total_principal + interest

    cross_year = ci.year_when_profit_exceeds_principal(times, balances, principals)
    if cross_year is not None:
        print(f"利润首次超过本金的时间：第 {cross_year:.2f} 年")
    else:
        print("在模拟期间内利润未超过本金")

    print(f"总本金 (本金+定投)：{total_principal:.2f}")
    print(f"总利润：{interest:.2f}")
    print(f"本金+利润：{final_amount:.2f}")
    print(f"最后一天的每日利息：{daily_interest:.4f}")

    # ===== 年度回报率统计 =====
    valid_start_balances = np.array(year_end_balances[:-1])
    valid_end_balances = np.array(year_end_balances[1:])
    valid_mask = valid_start_balances > 0
    annual_returns = (valid_end_balances[valid_mask] - valid_start_balances[valid_mask]) / valid_start_balances[valid_mask]
    mean_return = np.nanmean(annual_returns)
    std_return = np.nanstd(annual_returns)
    max_drawdown = min(drawdowns) if drawdowns else 0.0

    print(f"模拟年度回报率：平均 {mean_return * 100:.2f}%，标准差 {std_return * 100:.2f}%，最大回撤 {max_drawdown * 100:.1f}%")

    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    print(f"夏普比率：{sharpe_ratio:.2f}")

    # ===== 跳跃事件统计 =====
    if jump_magnitudes:
        jump_avg_drop = np.mean(jump_magnitudes) * 100
    else:
        jump_avg_drop = 0.0

    print(f"跳跃事件总次数：{jump_count} 次，平均跳幅 {jump_avg_drop:.1f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(times, balances, label='账户余额', color='navy')
    plt.plot(times, principals, label='累计本金', color='darkorange', linestyle='--')
    profits = balances - principals
    plt.plot(times, profits, label='累计利润', color='seagreen', linestyle='-.')
    final_time = times[-1]
    plt.text(final_time, balances[-1], f'{balances[-1]:.2f}', color='navy', fontsize=10, va='bottom', ha='right')
    plt.text(final_time, principals[-1], f'{principals[-1]:.2f}', color='darkorange', fontsize=10, va='bottom', ha='right')
    plt.text(final_time, profits[-1], f'{profits[-1]:.2f}', color='seagreen', fontsize=10, va='bottom', ha='right')

    plt.xlabel('年份')
    plt.ylabel('账户余额')
    plt.title('定投复利增长曲线（含费用与通胀）')
    plt.legend()
    plt.grid(True)
    plt.show()
