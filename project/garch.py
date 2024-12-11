import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

# 1. 数据准备：生成示例数据（可以替换为真实数据）
np.random.seed(42)
hours = 24 * 30  # 30天，每小时
prices = 50000 + np.cumsum(np.random.normal(0, 50, hours))  # 假设的价格路径
returns = np.log(prices[1:] / prices[:-1])  # 计算对数收益率

# 2. GARCH 模型拟合
model = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
result = model.fit(disp='off')
print(result.summary())

# 3. 波动性预测
forecast_horizon = 24  # 预测未来24小时
forecast = result.forecast(horizon=forecast_horizon)
cond_var = forecast.variance.iloc[-1].values  # 条件方差
predicted_volatility = np.sqrt(cond_var)  # 预测波动性

# 4. 蒙特卡洛模拟
N = 1000  # 模拟路径数量
initial_price = prices[-1]  # 当前价格
simulated_prices = np.zeros((N, forecast_horizon))

for i in range(N):
    simulated_returns = np.random.normal(0, predicted_volatility, forecast_horizon)
    simulated_path = [initial_price]
    for r in simulated_returns:
        next_price = simulated_path[-1] * np.exp(r)
        simulated_path.append(next_price)
    simulated_prices[i, :] = simulated_path[1:]

# 5. 可视化结果
plt.figure(figsize=(12, 6))
for i in range(min(N, 10)):  # 绘制前10条路径
    plt.plot(simulated_prices[i], alpha=0.7)
plt.title('Monte Carlo Simulations of Bitcoin Prices')
plt.xlabel('Hours')
plt.ylabel('Price')
plt.show()

# 6. 统计分析
mean_prices = np.mean(simulated_prices, axis=0)  # 平均路径
plt.plot(mean_prices, label='Mean Path', color='red')
plt.title('Mean Path of Simulated Bitcoin Prices')
plt.xlabel('Hours')
plt.ylabel('Price')
plt.legend()
plt.savefig("monte_carlo_simulation.png")  # 保存图像为 PNG 文件

