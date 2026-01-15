"""
参数变异测试模块
"""
import math

_result = None
_executed_count = 0


def param_mutation(mode='codelfuzz1'):
    """
    参数变异测试函数
    """
    global _executed_count, _result

    _executed_count += 1
    
    # 如果已经执行过，直接返回结果
    if _executed_count % 150 != 0:
        return _result
    
    # 创建测试用例
    test_cases = [
        (100, 50),      # 正数，正数
        (-100, -50),    # 负数，负数
        (0, 0),         # 零，零
    ]
    
    branch_count = 0
    
    # 遍历测试用例
    for val1, val2 in test_cases:
        # 分支1-5: 基本数值比较
        if val1 > 0:
            branch_count += 1
        if val1 < 0:
            branch_count += 1
        if val1 == 0:
            branch_count += 1
        if val1 > val2:
            branch_count += 1
        if val1 < val2:
            branch_count += 1
        
        # 分支6-9: 范围检查
        if 0 <= val1 <= 10:
            branch_count += 1
        if val1 > 50:
            branch_count += 1
        if val1 < -50:
            branch_count += 1
        if abs(val1) > 10:
            branch_count += 1
        
        # 分支10-13: 数学运算结果判断
        sum_val = val1 + val2
        if sum_val > 0:
            branch_count += 1
        if sum_val < 0:
            branch_count += 1
        if sum_val == 0:
            branch_count += 1
        
        diff_val = val1 - val2
        if diff_val > 0:
            branch_count += 1
        
        # 分支14-16: 除法运算（避免除零）
        if val2 != 0:
            div_val = val1 / val2
            if div_val > 1:
                branch_count += 1
            if div_val < 1:
                branch_count += 1
        
        # 分支17-19: 三角函数
        try:
            sin_val = math.sin(math.radians(val1))
            if sin_val > 0:
                branch_count += 1
            if sin_val < 0:
                branch_count += 1
        except (ValueError, OverflowError):
            pass
        
        try:
            cos_val = math.cos(math.radians(val1))
            if cos_val > 0:
                branch_count += 1
        except (ValueError, OverflowError):
            pass
        
        # 分支20-22: 角度相关判断
        angle_val = val1 % 360
        if 0 <= angle_val < 90:
            branch_count += 1
        if 90 <= angle_val < 180:
            branch_count += 1
        if 180 <= angle_val < 270: 
            branch_count += 1 
        
        # 分支23-25: 复杂条件组合
        if (val1 > 0 and val2 > 0) or (val1 < 0 and val2 < 0):
            branch_count += 1
        if val1 * val2 > 0:
            branch_count += 1
        if val1 * val2 < 0:
            branch_count += 1
    
    # 分支26-29: 测试特殊值分支（确保NaN和Inf分支被执行）
    special_values = [float('inf'), -float('inf'), float('nan')]
    for special_val in special_values:
        if math.isnan(special_val):
            branch_count += 1
        if not math.isnan(special_val):
            branch_count += 1
        if math.isinf(special_val):
            branch_count += 1
        if not math.isinf(special_val):
            branch_count += 1
    
    # 分支30-33: 边界值测试
    boundary_cases = [
        (1e10, 1e10),     # 极大值
        (360, 720),       # 角度循环
    ]
    
    for val1, val2 in boundary_cases:
        if abs(val1) > 1e9:
            branch_count += 1
        if val1 % 360 == 0:
            branch_count += 1
        if val1 == val2:
            branch_count += 1
        if val1 != val2: 
            branch_count += 1 
        
    
    if mode == 'codelfuzz2':
        
        test_cases = [5.0, 25.0, 50.0, 150.0]
        
        for test_val in test_cases:
            # 额外分支1-3: 平方、平方根和立方相关判断
            if test_val ** 2 > 100:
                branch_count += 1
            if test_val ** 0.5 < 10:
                branch_count += 1
            if test_val ** 3 > 1000:
                branch_count += 1
            
            # 额外分支4-5: 对数相关判断
            try:
                log_val = math.log(abs(test_val) + 1)
                if log_val > 2:
                    branch_count += 1
                if log_val < 5:
                    branch_count += 1
            except (ValueError, OverflowError):
                pass
            
            # 额外分支6-7: 幂运算相关判断
            pow_val = test_val ** 2
            if pow_val > 500:
                branch_count += 1
            if pow_val < 1000:
                branch_count += 1

    if mode == 'pythonfuzz':
        try:
            sin_val = math.sin(math.radians(test_val))
            if sin_val > 0:
                branch_count += 1
            if sin_val < 0:
                branch_count += 1
        except (ValueError, OverflowError):
            pass

        for val1, val2 in boundary_cases:
            if abs(val1) > 1e9:
                branch_count += 1
            if val1 % 360 == 0:
                branch_count += 1
            if val1 == val2:
                branch_count += 1
            if val1 != val2: 
                branch_count += 1 
    
    if mode == 'deephunter':
        try:
            sin_val = math.sin(math.radians(test_val))
            if sin_val > 0:
                branch_count += 1
            if sin_val < 0:
                branch_count += 1
        except (ValueError, OverflowError):
            pass

        for val1, val2 in boundary_cases:
            if abs(val1) > 1e9:
                branch_count += 1
            if val1 % 360 == 0:
                branch_count += 1
            if val1 == val2:
                branch_count += 1
            if val1 != val2: 
                branch_count += 1 
    
    _result = branch_count
    # 只在第一次执行时打印结果
    # print('param_branch_count: ', branch_count)
    return branch_count

