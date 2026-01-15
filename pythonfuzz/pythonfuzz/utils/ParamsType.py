import inspect

def get_parameter_types(func):
    if not func:
        return None
        
    sig = inspect.signature(func)
    param_types = {}
    
    # 如果只有一个参数且是 **kwargs，返回 None
    if len(sig.parameters) == 1 and list(sig.parameters.values())[0].kind == inspect.Parameter.VAR_KEYWORD:
        return None
        
    # 获取普通参数的类型
    for name, param in sig.parameters.items():
        # 跳过 **kwargs 参数
        if param.kind != inspect.Parameter.VAR_KEYWORD:
            param_types[name] = param.annotation if param.annotation != inspect._empty else None
            
    return param_types


