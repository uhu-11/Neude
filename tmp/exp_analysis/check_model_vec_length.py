"""
分析 obstacle_detection 和 traffic_light_detection 模型的内部结构和神经元向量长度
"""
import tensorflow as tf
import numpy as np
import os
import sys

# 添加必要的路径
sys.path.insert(0, '/media/lzq/D/lzq/pylot_test/pylot')
sys.path.insert(0, '/media/lzq/D/lzq/pylot_test/pythonfuzz')

def analyze_model_structure(model_path, model_name):
    """分析模型的结构和神经元向量长度"""
    print("=" * 80)
    print(f"分析模型: {model_name}")
    print(f"模型路径: {model_path}")
    print("=" * 80)
    
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return None
    
    try:
        # 加载模型
        print("\n正在加载模型...")
        model = tf.saved_model.load(model_path)
        
        # 获取模型的signature
        print("\n可用的signatures:")
        for sig_name in model.signatures.keys():
            print(f"  - {sig_name}")
        
        # 使用serving_default signature
        if 'serving_default' in model.signatures:
            infer_func = model.signatures['serving_default']
            
            # 获取输入输出信息
            print("\n输入信息:")
            for input_name, input_spec in infer_func.structured_input_signature[1].items():
                print(f"  {input_name}: shape={input_spec.shape}, dtype={input_spec.dtype}")
            
            print("\n输出信息:")
            for output_name, output_spec in infer_func.structured_outputs.items():
                print(f"  {output_name}: shape={output_spec.shape}, dtype={output_spec.dtype}")
        
        # 尝试获取模型的内部结构（如果可能）
        print("\n尝试获取模型内部结构...")
        
        # 方法1: 如果是Keras模型，可以获取层信息
        try:
            # 尝试将SavedModel转换为Keras模型
            keras_model = tf.keras.models.load_model(model_path)
            print("\n模型类型: Keras Model")
            print(f"总层数: {len(keras_model.layers)}")
            print("\n模型层结构:")
            keras_model.summary()
            
            # 获取每层的参数数量
            print("\n各层参数统计:")
            total_params = 0
            for i, layer in enumerate(keras_model.layers):
                params = layer.count_params()
                total_params += params
                if params > 0:  # 只显示有参数的层
                    print(f"  Layer {i} ({layer.__class__.__name__}): {params:,} 参数")
                    if hasattr(layer, 'output_shape'):
                        print(f"    输出形状: {layer.output_shape}")
            print(f"\n总参数数: {total_params:,}")
            
        except Exception as e:
            print(f"无法转换为Keras模型: {e}")
            print("模型可能是纯TensorFlow SavedModel格式")
        
        # 方法2: 尝试获取计算图中的节点信息
        try:
            print("\n尝试分析计算图...")
            graph = infer_func.graph
            print(f"计算图节点数: {len(list(graph.as_graph_def().node))}")
            
            # 获取所有操作节点
            operations = graph.get_operations()
            print(f"操作节点数: {len(operations)}")
            
            # 统计不同类型的操作
            op_types = {}
            for op in operations[:100]:  # 限制显示前100个
                op_type = op.type
                op_types[op_type] = op_types.get(op_type, 0) + 1
            
            print("\n主要操作类型统计（前20个）:")
            sorted_ops = sorted(op_types.items(), key=lambda x: x[1], reverse=True)[:20]
            for op_type, count in sorted_ops:
                print(f"  {op_type}: {count}")
                
        except Exception as e:
            print(f"无法分析计算图: {e}")
        
        # 方法3: 创建测试输入，获取中间层输出（神经元向量）
        print("\n" + "-" * 80)
        print("分析神经元向量长度")
        print("-" * 80)
        
        # 创建测试输入（随机图像）
        test_input = np.random.randint(0, 255, size=(1, 1080, 1920, 3), dtype=np.uint8)
        test_tensor = tf.convert_to_tensor(test_input, dtype=tf.uint8)
        
        # 运行模型获取输出
        print("\n运行模型推理...")
        result = infer_func(test_tensor)
        
        print("\n输出张量的形状和大小:")
        for output_name, output_tensor in result.items():
            shape = output_tensor.shape
            try:
                size = tf.size(output_tensor).numpy()
            except:
                # 如果无法直接调用numpy()，尝试其他方法
                size = np.prod(output_tensor.shape) if output_tensor.shape.is_fully_defined() else 0
            print(f"  {output_name}:")
            print(f"    形状: {shape}")
            print(f"    总元素数: {size:,}")
            print(f"    数据类型: {output_tensor.dtype}")
            
            # 如果是多维张量，计算每个维度的长度
            try:
                if shape.ndims is not None and shape.ndims > 1:
                    dims = [dim if dim is not None else '?' for dim in shape.as_list()]
                    print(f"    各维度长度: {dims}")
            except:
                pass
        
        return model, result
        
    except Exception as e:
        print(f"分析模型时出错: {e}")
        import traceback
        traceback.print_exc()
        # 即使出错也返回部分结果
        return model, None

def count_model_parameters(model_path, model_name):
    """统计模型的总参数数量（神经元数量）"""
    print("\n" + "=" * 80)
    print(f"统计模型参数数量: {model_name}")
    print("=" * 80)
    
    try:
        model = tf.saved_model.load(model_path)
        
        total_params = 0
        trainable_params = 0
        variable_info = []
        
        # 方法1: 尝试从model.variables获取（SavedModel v2）
        try:
            if hasattr(model, 'variables'):
                variables = list(model.variables)
                if len(variables) > 0:
                    print(f"\n方法1: 从model.variables找到 {len(variables)} 个变量")
                    for var in variables:
                        try:
                            var_shape = var.shape
                            var_size = np.prod(var_shape) if var_shape.is_fully_defined() else 0
                            if var_size > 0:
                                var_dtype = var.dtype
                                is_trainable = getattr(var, 'trainable', True)
                                
                                total_params += var_size
                                if is_trainable:
                                    trainable_params += var_size
                                
                                variable_info.append({
                                    'name': var.name,
                                    'shape': var_shape,
                                    'size': var_size,
                                    'dtype': str(var_dtype),
                                    'trainable': is_trainable
                                })
                        except Exception as e:
                            pass
        except Exception as e:
            print(f"方法1失败: {e}")
        
        # 方法2: 从计算图中提取所有变量节点
        if total_params == 0:
            try:
                infer_func = model.signatures['serving_default']
                graph = infer_func.graph
                
                print(f"\n方法2: 分析计算图...")
                print(f"计算图节点总数: {len(list(graph.get_operations()))}")
                
                # 获取所有操作节点
                all_ops = graph.get_operations()
                
                # 查找变量相关的操作
                var_ops = []
                for op in all_ops:
                    op_type = op.type
                    # 查找变量相关的操作类型
                    if any(keyword in op_type for keyword in ['Variable', 'VarHandle', 'ReadVariable', 'Assign']):
                        var_ops.append(op)
                
                print(f"找到 {len(var_ops)} 个变量相关操作")
                
                # 尝试从ReadVariableOp获取变量形状
                for op in var_ops:
                    try:
                        # 获取操作的输出
                        if len(op.outputs) > 0:
                            output = op.outputs[0]
                            if output.shape.is_fully_defined():
                                var_size = np.prod(output.shape.as_list())
                                if var_size > 0:
                                    total_params += var_size
                                    variable_info.append({
                                        'name': op.name,
                                        'shape': output.shape,
                                        'size': var_size,
                                        'dtype': str(output.dtype),
                                        'trainable': True
                                    })
                    except:
                        pass
                
                # 如果还是没找到，尝试遍历所有Const节点（可能包含权重）
                if total_params == 0:
                    print("\n方法3: 尝试从Const节点提取权重信息...")
                    const_ops = [op for op in all_ops if op.type == 'Const']
                    print(f"找到 {len(const_ops)} 个Const节点")
                    
                    # Const节点通常包含权重，但数量可能很大，只统计较大的
                    for op in const_ops:
                        try:
                            if len(op.outputs) > 0:
                                output = op.outputs[0]
                                if output.shape.is_fully_defined():
                                    var_size = np.prod(output.shape.as_list())
                                    # 只统计较大的权重（可能是模型参数）
                                    if var_size > 100:  # 阈值，过滤小的常量
                                        total_params += var_size
                                        variable_info.append({
                                            'name': op.name,
                                            'shape': output.shape,
                                            'size': var_size,
                                            'dtype': str(output.dtype),
                                            'trainable': False  # Const通常不可训练
                                        })
                        except:
                            pass
                            
            except Exception as e:
                print(f"方法2失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 显示变量信息（限制显示数量）
        if len(variable_info) > 0:
            print(f"\n变量详情（显示前30个，共{len(variable_info)}个）:")
            for i, var_info in enumerate(variable_info[:30]):
                print(f"  [{i+1}] {var_info['name']}")
                print(f"      形状: {var_info['shape']}")
                print(f"      参数数量: {var_info['size']:,}")
                print(f"      数据类型: {var_info['dtype']}")
                print(f"      可训练: {var_info['trainable']}")
            if len(variable_info) > 30:
                print(f"  ... 还有 {len(variable_info) - 30} 个变量未显示")
        
        # 按类型分组统计
        if len(variable_info) > 0:
            print("\n按变量名称前缀分组统计:")
            prefix_stats = {}
            for var_info in variable_info:
                # 提取变量名称的前缀（通常是层名）
                name_parts = var_info['name'].split('/')
                if len(name_parts) > 0:
                    prefix = name_parts[0] if len(name_parts) > 1 else 'other'
                    if prefix not in prefix_stats:
                        prefix_stats[prefix] = {'count': 0, 'total_size': 0}
                    prefix_stats[prefix]['count'] += 1
                    prefix_stats[prefix]['total_size'] += var_info['size']
            
            sorted_prefixes = sorted(prefix_stats.items(), key=lambda x: x[1]['total_size'], reverse=True)
            print(f"{'前缀':<40} {'变量数':<15} {'参数数量':<20}")
            print("-" * 75)
            for prefix, stats in sorted_prefixes[:20]:  # 显示前20个
                print(f"{prefix[:40]:<40} {stats['count']:<15} {stats['total_size']:>20,}")
        
        print("\n" + "-" * 80)
        print("参数统计总结:")
        print("-" * 80)
        print(f"总参数数量（神经元数量）: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"不可训练参数数量: {total_params - trainable_params:,}")
        print(f"变量总数: {len(variable_info)}")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'variable_info': variable_info,
            'variable_count': len(variable_info)
        }
        
    except Exception as e:
        print(f"统计参数时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_all_conv_layers(model_path, model_name, filter_duplicates=True):
    """查找模型中所有卷积层和全连接层
    
    Args:
        model_path: 模型路径
        model_name: 模型名称
        filter_duplicates: 如果为True，只统计主版本（无后缀）的层，过滤掉_1, _2等重复版本
    """
    print("\n" + "=" * 80)
    print(f"查找所有可用的中间层: {model_name}")
    print("=" * 80)
    
    frozen_graph_path = os.path.dirname(model_path.rstrip('/'))
    frozen_graph_path = os.path.join(frozen_graph_path, 'frozen_graph.pb')
    
    if not os.path.exists(frozen_graph_path):
        print(f"警告: 未找到frozen_graph.pb文件: {frozen_graph_path}")
        return []
    
    all_layers = []
    
    try:
        tf.compat.v1.disable_eager_execution()
        with tf.compat.v1.Session() as sess:
            tf.compat.v1.global_variables_initializer().run()
            output_graph_def = tf.compat.v1.GraphDef()
            
            with open(frozen_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")
            
            graph = tf.compat.v1.get_default_graph()
            all_ops = graph.get_operations()
            
            # 查找所有中间层 - 通过检查操作的输出形状
            conv_layers = []
            matmul_layers = []
            seen_names = set()
            # 添加基于层特征的去重：使用规范化层名称作为唯一标识
            # 规范化：移除可能的重复前缀（如resnet_v1_101/resnet_v1_101 -> resnet_v1_101）
            seen_normalized_names = set()
            
            for op in all_ops:
                op_name = op.name
                op_type = op.type
                
                # 跳过某些不需要的操作
                if op_type in ['Const', 'Placeholder', 'VariableV2', 'Assign', 'Identity']:
                    continue
                
                # 检查操作的输出
                try:
                    if len(op.outputs) > 0:
                        for i, output in enumerate(op.outputs):
                            output_name = f"{op_name}:{i}"
                            if output_name in seen_names:
                                continue
                            
                            shape = output.shape
                            if shape.ndims is None:
                                continue
                            
                            # 查找4维输出（卷积层）
                            if shape.ndims == 4:
                                try:
                                    channels = shape[-1].value if hasattr(shape[-1], 'value') else shape[-1]
                                    if channels is not None and channels > 0:
                                        # 只保留真正的Conv2D操作输出，过滤掉BatchNorm、Relu、add等
                                        if (op_type == 'Conv2D' or 'Conv2D' in op_name) and channels >= 16:
                                            # 进一步过滤：排除BatchNorm、Relu、add等操作
                                            exclude_keywords = ['BatchNorm', 'Relu', 'add', 'BiasAdd', 'Mul', 'Add', 'Reshape', 'MaxPool', 'AvgPool', 'CropAndResize']
                                            if not any(keyword in op_name for keyword in exclude_keywords):
                                                # 规范化层名称：移除重复的前缀部分
                                                # 例如：resnet_v1_101/resnet_v1_101/block1 -> resnet_v1_101/block1
                                                normalized_name = op_name
                                                # 移除重复的路径段（如果存在）
                                                name_parts = normalized_name.split('/')
                                                # 查找并移除连续的重复段
                                                cleaned_parts = []
                                                prev_part = None
                                                for part in name_parts:
                                                    if part != prev_part:
                                                        cleaned_parts.append(part)
                                                    prev_part = part
                                                normalized_name = '/'.join(cleaned_parts)
                                                
                                                # 如果启用过滤重复版本，只保留主版本（无后缀或后缀为_0）
                                                if filter_duplicates:
                                                    import re
                                                    # 检查是否有数字后缀（如Conv2D_1, Conv2D_2）
                                                    if re.search(r'_\d+(:0)?$', normalized_name):
                                                        # 跳过带后缀的版本
                                                        continue
                                                
                                                # 创建唯一标识：使用规范化名称和通道数
                                                layer_key = (normalized_name, int(channels))
                                                
                                                # 如果这个层已经见过，跳过（避免重复统计）
                                                if layer_key in seen_normalized_names:
                                                    continue
                                                
                                                seen_names.add(output_name)
                                                seen_normalized_names.add(layer_key)
                                                conv_layers.append({
                                                    'name': output_name,
                                                    'shape': shape,
                                                    'channels': int(channels),
                                                    'type': 'Conv2D',
                                                    'op_type': op_type
                                                })
                                except:
                                    pass
                            
                            # 查找2维输出（全连接层）
                            elif shape.ndims == 2:
                                try:
                                    features = shape[-1].value if hasattr(shape[-1], 'value') else shape[-1]
                                    if features is not None and features > 0:
                                        if features >= 8:  # 只保留特征数>=8的层
                                            # 规范化层名称
                                            normalized_name = op_name
                                            name_parts = normalized_name.split('/')
                                            cleaned_parts = []
                                            prev_part = None
                                            for part in name_parts:
                                                if part != prev_part:
                                                    cleaned_parts.append(part)
                                                prev_part = part
                                            normalized_name = '/'.join(cleaned_parts)
                                            
                                            # 如果启用过滤重复版本，只保留主版本
                                            if filter_duplicates:
                                                import re
                                                if re.search(r'_\d+(:0)?$', normalized_name):
                                                    continue
                                            
                                            layer_key = (normalized_name, int(features))
                                            
                                            if layer_key in seen_normalized_names:
                                                continue
                                            
                                            seen_names.add(output_name)
                                            seen_normalized_names.add(layer_key)
                                            matmul_layers.append({
                                                'name': output_name,
                                                'shape': shape,
                                                'channels': int(features),
                                                'type': 'MatMul',
                                                'op_type': op_type
                                            })
                                except:
                                    pass
                except:
                    pass
            
            # 查找所有MatMul操作（全连接层）- 注意：这里会覆盖之前的matmul_layers
            # 但我们已经在上面的循环中处理了MatMul，所以这里可能不需要
            # 保留这部分代码以防遗漏，但添加去重检查
            additional_matmul = []
            for op in all_ops:
                op_name = op.name
                if op_name in seen_names:
                    continue
                if ('MatMul' in op_name or 'matmul' in op_name.lower()) and op_name.endswith(':0'):
                    seen_names.add(op_name)
                    try:
                        tensor = graph.get_tensor_by_name(op_name)
                        shape = tensor.shape
                        if shape.ndims == 2:  # 全连接层输出是2维的
                            features = shape[-1].value if hasattr(shape[-1], 'value') else shape[-1]
                            if features is not None and features > 0:
                                # 规范化层名称
                                normalized_name = op_name
                                name_parts = normalized_name.split('/')
                                cleaned_parts = []
                                prev_part = None
                                for part in name_parts:
                                    if part != prev_part:
                                        cleaned_parts.append(part)
                                    prev_part = part
                                normalized_name = '/'.join(cleaned_parts)
                                
                                layer_key = (normalized_name, int(features))
                                
                                if layer_key not in seen_normalized_names:
                                    seen_normalized_names.add(layer_key)
                                    additional_matmul.append({
                                        'name': op_name,
                                        'shape': shape,
                                        'channels': int(features),
                                        'type': 'MatMul'
                                    })
                    except:
                        try:
                            if len(op.outputs) > 0:
                                output = op.outputs[0]
                                if output.shape.ndims == 2:
                                    features = output.shape[-1].value if hasattr(output.shape[-1], 'value') else output.shape[-1]
                                    if features is not None and features > 0:
                                        # 规范化层名称
                                        normalized_name = op_name
                                        name_parts = normalized_name.split('/')
                                        cleaned_parts = []
                                        prev_part = None
                                        for part in name_parts:
                                            if part != prev_part:
                                                cleaned_parts.append(part)
                                            prev_part = part
                                        normalized_name = '/'.join(cleaned_parts)
                                        
                                        layer_key = (normalized_name, int(features))
                                        
                                        if layer_key not in seen_normalized_names:
                                            seen_normalized_names.add(layer_key)
                                            additional_matmul.append({
                                                'name': op_name,
                                                'shape': output.shape,
                                                'channels': int(features),
                                                'type': 'MatMul'
                                            })
                        except:
                            pass
            
            # 合并所有层
            all_layers = conv_layers + matmul_layers + additional_matmul
            
            # 调试信息：显示去重统计
            print(f"\n调试信息:")
            print(f"  总操作数: {len(all_ops)}")
            print(f"  找到的Conv2D层: {len(conv_layers)}")
            print(f"  找到的MatMul层（第一遍）: {len(matmul_layers)}")
            print(f"  找到的MatMul层（第二遍）: {len(additional_matmul)}")
            print(f"  唯一规范化层数: {len(seen_normalized_names)}")
            print(f"  最终总层数: {len(all_layers)}")
            
            # 检查是否有重复的网络结构（通过检查层名称模式）
            # 如果发现大量带_1, _2等后缀的层，可能包含多个网络副本
            suffix_patterns = {}
            for layer in all_layers:
                name = layer['name']
                # 检查是否有数字后缀（如Conv2D_1, Conv2D_2）
                import re
                match = re.search(r'(_\d+)(:0)?$', name)
                if match:
                    suffix = match.group(1)
                    base_name = name.replace(suffix, '')
                    if base_name not in suffix_patterns:
                        suffix_patterns[base_name] = []
                    if suffix not in suffix_patterns[base_name]:
                        suffix_patterns[base_name].append(suffix)
            
            duplicate_bases = {k: v for k, v in suffix_patterns.items() if len(v) > 1}
            if duplicate_bases:
                print(f"\n⚠️  警告: 发现可能的重复网络结构！")
                print(f"  发现 {len(duplicate_bases)} 个层有多个版本（带不同后缀）")
                print(f"  这可能意味着frozen_graph.pb中包含了多个网络副本")
                print(f"  示例（前3个）:")
                for i, (base, suffixes) in enumerate(list(duplicate_bases.items())[:3]):
                    print(f"    {base[:60]}... 有版本: {suffixes}")
                print(f"\n  建议: 只统计主版本（无后缀或后缀为_0）的层，或检查模型文件")
            
            sess.close()
            
    except Exception as e:
        print(f"查找中间层时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return all_layers

def analyze_nac_vector_structure(model_path, model_name):
    """分析NAC（神经元激活覆盖）向量的结构和长度"""
    print("\n" + "=" * 80)
    print(f"分析NAC向量结构: {model_name}")
    print("=" * 80)
    
    # NAC覆盖使用的10个中间层（来自neural_cov.py）
    tensor_names = [
        'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block1/unit_1/bottleneck_v1/shortcut/Conv2D:0',
        'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block1/unit_1/bottleneck_v1/conv1/Conv2D:0',
        'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block1/unit_2/bottleneck_v1/conv1/Conv2D:0',
        'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block3/unit_20/bottleneck_v1/conv2/Conv2D:0',
        'FirstStageBoxPredictor/ClassPredictor/Conv2D:0',
        'FirstStageBoxPredictor/BoxEncodingPredictor/Conv2D:0',
        'SecondStageFeatureExtractor/resnet_v1_101/block4/unit_1/bottleneck_v1/conv1/Conv2D:0',
        'SecondStageFeatureExtractor/resnet_v1_101/block4/unit_2/bottleneck_v1/conv1/Conv2D:0',
        'SecondStageFeatureExtractor/resnet_v1_101/block4/unit_3/bottleneck_v1/conv2/Conv2D:0',
        'SecondStageFeatureExtractor/resnet_v1_101/block4/unit_3/bottleneck_v1/conv3/Conv2D:0'
    ]
    
    # 尝试从frozen_graph.pb文件读取
    # 构建可能的路径
    base_dir = os.path.dirname(model_path.rstrip('/'))
    frozen_graph_path = os.path.join(base_dir, 'frozen_graph.pb')
    
    if not os.path.exists(frozen_graph_path):
        # 尝试其他可能的路径
        alternative_paths = [
            model_path.replace('faster-rcnn/', 'frozen_graph.pb'),
            model_path.replace('/faster-rcnn', '/frozen_graph.pb'),
            os.path.join(os.path.dirname(model_path), 'frozen_graph.pb'),
        ]
        for path in alternative_paths:
            if os.path.exists(path):
                frozen_graph_path = path
                break
        else:
            print(f"\n警告: 未找到frozen_graph.pb文件")
            print(f"尝试过的路径: {[frozen_graph_path] + alternative_paths}")
            frozen_graph_path = None
    
    nac_layers_info = []
    total_nac_length = 0
    
    try:
        if frozen_graph_path and os.path.exists(frozen_graph_path):
            # 方法1: 从frozen_graph.pb读取
            print(f"\n从frozen_graph.pb读取: {frozen_graph_path}")
            tf.compat.v1.disable_eager_execution()
            with tf.compat.v1.Session() as sess:
                tf.compat.v1.global_variables_initializer().run()
                output_graph_def = tf.compat.v1.GraphDef()
                
                with open(frozen_graph_path, "rb") as f:
                    output_graph_def.ParseFromString(f.read())
                    _ = tf.import_graph_def(output_graph_def, name="")
                
                graph = tf.compat.v1.get_default_graph()
                
                print("\nNAC覆盖使用的中间层信息:")
                print(f"{'序号':<5} {'层名称':<80} {'输出形状':<30} {'通道数':<10} {'累计长度':<10}")
                print("-" * 140)
                
                for idx, tensor_name in enumerate(tensor_names, 1):
                    try:
                        tensor = graph.get_tensor_by_name(tensor_name)
                        shape = tensor.shape
                        # 获取通道数（最后一维）
                        if shape.ndims > 0:
                            channels = shape[-1].value if hasattr(shape[-1], 'value') else shape[-1]
                            if channels is None:
                                # 尝试从shape推断
                                if len(shape) == 4:  # (batch, height, width, channels)
                                    channels = shape[3].value if hasattr(shape[3], 'value') else shape[3]
                            
                            if channels is not None:
                                total_nac_length += int(channels)
                                nac_layers_info.append({
                                    'index': idx,
                                    'name': tensor_name,
                                    'shape': shape,
                                    'channels': int(channels),
                                    'cumulative_length': total_nac_length
                                })
                                
                                print(f"{idx:<5} {tensor_name[:80]:<80} {str(shape):<30} {int(channels):<10} {total_nac_length:<10}")
                            else:
                                print(f"{idx:<5} {tensor_name[:80]:<80} {str(shape):<30} {'未知':<10} {'未知':<10}")
                        else:
                            print(f"{idx:<5} {tensor_name[:80]:<80} {'未知形状':<30} {'未知':<10} {'未知':<10}")
                    except Exception as e:
                        print(f"{idx:<5} {tensor_name[:80]:<80} {'获取失败':<30} {'错误':<10} {str(e)[:30]}")
                
                # 不需要reset，session会自动清理
                sess.close()
        else:
            # 方法2: 尝试从SavedModel提取（需要不同的方法）
            print(f"\n尝试从SavedModel提取中间层信息...")
            print("注意: SavedModel格式可能无法直接访问中间层，需要实际运行推理")
            
            # 这里可以尝试加载模型并运行推理来获取中间层输出
            # 但需要更复杂的实现
            
    except Exception as e:
        print(f"\n分析NAC向量结构时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-" * 80)
    print("NAC向量结构总结:")
    print("-" * 80)
    print(f"中间层总数: {len(tensor_names)}")
    print(f"总NAC向量长度: {total_nac_length}")
    print(f"各层通道数: {[info['channels'] for info in nac_layers_info]}")
    
    if total_nac_length > 0:
        print(f"\n各层对NAC向量长度的贡献:")
        print(f"{'层序号':<8} {'通道数':<10} {'占比':<10} {'累计占比':<10}")
        print("-" * 40)
        cumulative = 0
        for info in nac_layers_info:
            cumulative += info['channels']
            percentage = (info['channels'] / total_nac_length) * 100
            cum_percentage = (cumulative / total_nac_length) * 100
            print(f"{info['index']:<8} {info['channels']:<10} {percentage:>6.2f}%   {cum_percentage:>6.2f}%")
    
    return {
        'total_length': total_nac_length,
        'layers_info': nac_layers_info,
        'layer_count': len(tensor_names)
    }

def analyze_all_available_layers(model_path, model_name, filter_duplicates=True):
    """分析模型中所有可用的中间层
    
    Args:
        model_path: 模型路径
        model_name: 模型名称
        filter_duplicates: 如果为True，只统计主版本（无后缀）的层
    """
    print("\n" + "=" * 80)
    print(f"分析所有可用的中间层: {model_name}")
    print("=" * 80)
    
    all_layers = find_all_conv_layers(model_path, model_name, filter_duplicates=filter_duplicates)
    
    if len(all_layers) == 0:
        print("未找到任何中间层")
        return None
    
    # 按通道数排序
    all_layers_sorted = sorted(all_layers, key=lambda x: x['channels'], reverse=True)
    
    print(f"\n找到 {len(all_layers)} 个中间层（Conv2D和MatMul）")
    print(f"\n{'序号':<6} {'层类型':<10} {'层名称':<80} {'输出形状':<30} {'通道数/特征数':<15} {'累计长度':<12}")
    print("-" * 160)
    
    total_channels = 0
    layers_info = []
    for idx, layer in enumerate(all_layers_sorted, 1):
        total_channels += layer['channels']
        layers_info.append({
            'index': idx,
            'name': layer['name'],
            'type': layer['type'],
            'shape': layer['shape'],
            'channels': layer['channels'],
            'cumulative_length': total_channels
        })
        
        layer_short_name = layer['name'][:80]
        print(f"{idx:<6} {layer['type']:<10} {layer_short_name:<80} {str(layer['shape']):<30} {layer['channels']:<15} {total_channels:<12}")
    
    # 统计信息
    conv_count = sum(1 for l in all_layers if l['type'] == 'Conv2D')
    matmul_count = sum(1 for l in all_layers if l['type'] == 'MatMul')
    
    print("\n" + "-" * 80)
    print("统计信息:")
    print("-" * 80)
    print(f"总层数: {len(all_layers)}")
    print(f"  - Conv2D层: {conv_count}")
    print(f"  - MatMul层（全连接）: {matmul_count}")
    print(f"总通道数/特征数: {total_channels:,}")
    print(f"平均每层通道数: {total_channels / len(all_layers):.2f}")
    
    # 按类型分组统计
    print("\n按类型分组统计:")
    print(f"{'类型':<15} {'层数':<10} {'总通道数':<15} {'平均通道数':<15}")
    print("-" * 55)
    
    if conv_count > 0:
        conv_channels = sum(l['channels'] for l in all_layers if l['type'] == 'Conv2D')
        print(f"{'Conv2D':<15} {conv_count:<10} {conv_channels:<15} {conv_channels/conv_count:<15.2f}")
    
    if matmul_count > 0:
        matmul_channels = sum(l['channels'] for l in all_layers if l['type'] == 'MatMul')
        print(f"{'MatMul':<15} {matmul_count:<10} {matmul_channels:<15} {matmul_channels/matmul_count:<15.2f}")
    
    # 找出最大的几层
    print("\n通道数最多的前10层:")
    print(f"{'序号':<6} {'层类型':<10} {'层名称（简化）':<60} {'通道数':<12}")
    print("-" * 90)
    for layer in all_layers_sorted[:10]:
        short_name = '/'.join(layer['name'].split('/')[-3:])[:60]
        print(f"{all_layers_sorted.index(layer)+1:<6} {layer['type']:<10} {short_name:<60} {layer['channels']:<12}")
    
    return {
        'total_layers': len(all_layers),
        'total_channels': total_channels,
        'layers_info': layers_info,
        'conv_count': conv_count,
        'matmul_count': matmul_count
    }

def extract_intermediate_layers(model_path, model_name):
    """尝试提取中间层的神经元向量"""
    print("\n" + "=" * 80)
    print(f"尝试提取中间层神经元向量: {model_name}")
    print("=" * 80)
    
    try:
        model = tf.saved_model.load(model_path)
        infer_func = model.signatures['serving_default']
        
        # 创建测试输入
        test_input = np.random.randint(0, 255, size=(1, 1080, 1920, 3), dtype=np.uint8)
        test_tensor = tf.convert_to_tensor(test_input, dtype=tf.uint8)
        
        # 如果是Keras模型，可以获取中间层输出
        try:
            keras_model = tf.keras.models.load_model(model_path)
            
            print("\n提取中间层输出（神经元向量）:")
            layer_outputs = {}
            
            # 获取所有层的输出
            for i, layer in enumerate(keras_model.layers):
                if hasattr(layer, 'output'):
                    try:
                        # 创建中间模型来获取该层的输出
                        intermediate_model = tf.keras.Model(
                            inputs=keras_model.input,
                            outputs=layer.output
                        )
                        
                        # 转换输入格式（如果需要）
                        if keras_model.input.shape[1:] != test_input.shape[1:]:
                            # 调整输入大小
                            target_shape = keras_model.input.shape[1:]
                            if None not in target_shape:
                                resized_input = tf.image.resize(
                                    tf.cast(test_input, tf.float32),
                                    target_shape[:2]
                                )
                            else:
                                resized_input = tf.cast(test_input, tf.float32)
                        else:
                            resized_input = tf.cast(test_input, tf.float32)
                        
                        # 获取该层的输出
                        layer_output = intermediate_model(resized_input)
                        
                        output_shape = layer_output.shape
                        output_size = tf.size(layer_output).numpy()
                        
                        layer_outputs[layer.name] = {
                            'shape': output_shape,
                            'size': output_size,
                            'dtype': layer_output.dtype
                        }
                        
                        print(f"\n  Layer {i}: {layer.name} ({layer.__class__.__name__})")
                        print(f"    输出形状: {output_shape}")
                        print(f"    神经元向量长度: {output_size:,}")
                        
                        # 如果是特征向量（通常是最后一层卷积或全连接层之前）
                        if len(output_shape) == 2:  # (batch, features)
                            print(f"    特征向量维度: {output_shape[1]}")
                        elif len(output_shape) == 4:  # (batch, height, width, channels)
                            print(f"    特征图: {output_shape[1]}x{output_shape[2]}x{output_shape[3]}")
                            
                    except Exception as e:
                        # 跳过无法提取的层
                        pass
                        
            return layer_outputs
            
        except Exception as e:
            print(f"无法提取中间层（非Keras模型）: {e}")
            return None
            
    except Exception as e:
        print(f"提取中间层时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # 模型路径配置
    obstacle_model_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/obstacle_detection/faster-rcnn/'
    traffic_light_model_path = '/media/lzq/D/lzq/pylot_test/pylot/dependencies/models/traffic_light_detection/faster-rcnn/'
    
    # 分析障碍物检测模型
    obstacle_model, obstacle_result = analyze_model_structure(
        obstacle_model_path, 
        "Obstacle Detection (Faster R-CNN)"
    )
    
    # 统计障碍物检测模型的参数数量
    obstacle_params = count_model_parameters(
        obstacle_model_path,
        "Obstacle Detection"
    )
    
    # 分析障碍物检测模型的NAC向量结构
    obstacle_nac_info = analyze_nac_vector_structure(
        obstacle_model_path,
        "Obstacle Detection"
    )
    
    # 分析障碍物检测模型中所有可用的中间层（过滤重复版本）
    obstacle_all_layers_info = analyze_all_available_layers(
        obstacle_model_path,
        "Obstacle Detection",
        filter_duplicates=True
    )
    
    if obstacle_model:
        obstacle_layers = extract_intermediate_layers(
            obstacle_model_path,
            "Obstacle Detection"
        )
    
    print("\n\n")
    
    # 分析红绿灯检测模型
    traffic_light_model, traffic_light_result = analyze_model_structure(
        traffic_light_model_path,
        "Traffic Light Detection (Faster R-CNN)"
    )
    
    # 统计红绿灯检测模型的参数数量
    traffic_light_params = count_model_parameters(
        traffic_light_model_path,
        "Traffic Light Detection"
    )
    
    # 分析红绿灯检测模型的NAC向量结构
    traffic_light_nac_info = analyze_nac_vector_structure(
        traffic_light_model_path,
        "Traffic Light Detection"
    )
    
    # 分析红绿灯检测模型中所有可用的中间层（过滤重复版本）
    traffic_light_all_layers_info = analyze_all_available_layers(
        traffic_light_model_path,
        "Traffic Light Detection",
        filter_duplicates=True
    )
    
    if traffic_light_model:
        traffic_light_layers = extract_intermediate_layers(
            traffic_light_model_path,
            "Traffic Light Detection"
        )
    
    # 对比总结
    print("\n" + "=" * 80)
    print("模型对比总结")
    print("=" * 80)
    
    if obstacle_result and traffic_light_result:
        print("\n输出张量对比:")
        print(f"{'输出名称':<30} {'障碍物检测形状':<30} {'红绿灯检测形状':<30}")
        print("-" * 90)
        
        # 获取所有输出名称
        all_outputs = set(obstacle_result.keys()) | set(traffic_light_result.keys())
        for output_name in sorted(all_outputs):
            obstacle_shape = str(obstacle_result.get(output_name, {}).shape) if output_name in obstacle_result else "N/A"
            traffic_shape = str(traffic_light_result.get(output_name, {}).shape) if output_name in traffic_light_result else "N/A"
            print(f"{output_name:<30} {obstacle_shape:<30} {traffic_shape:<30}")
    
    # 参数数量对比
    print("\n" + "=" * 80)
    print("模型参数数量对比")
    print("=" * 80)
    
    if obstacle_params and traffic_light_params:
        print(f"\n{'指标':<30} {'障碍物检测模型':<30} {'红绿灯检测模型':<30} {'差异':<30}")
        print("-" * 120)
        
        obs_total = obstacle_params.get('total_params', 0)
        obs_trainable = obstacle_params.get('trainable_params', 0)
        tl_total = traffic_light_params.get('total_params', 0)
        tl_trainable = traffic_light_params.get('trainable_params', 0)
        
        print(f"{'总参数数量':<30} {obs_total:>30,} {tl_total:>30,} {obs_total - tl_total:>30,}")
        print(f"{'可训练参数数量':<30} {obs_trainable:>30,} {tl_trainable:>30,} {obs_trainable - tl_trainable:>30,}")
        print(f"{'不可训练参数数量':<30} {obs_total - obs_trainable:>30,} {tl_total - tl_trainable:>30,} {(obs_total - obs_trainable) - (tl_total - tl_trainable):>30,}")
        
        if obs_total > 0 and tl_total > 0:
            diff_percent = ((obs_total - tl_total) / obs_total) * 100
            print(f"\n参数数量差异百分比: {diff_percent:.2f}%")
            
            print("\n主要区别:")
            if obs_total > tl_total:
                print(f"  障碍物检测模型比红绿灯检测模型多 {obs_total - tl_total:,} 个参数")
                print(f"  这可能是因为障碍物检测需要识别更多类别的对象")
            elif tl_total > obs_total:
                print(f"  红绿灯检测模型比障碍物检测模型多 {tl_total - obs_total:,} 个参数")
                print(f"  这可能是因为红绿灯检测需要更精细的特征提取")
            else:
                print("  两个模型的参数数量相同")
    else:
        print("\n无法获取完整的参数统计信息")
    
    # NAC向量结构对比
    print("\n" + "=" * 80)
    print("NAC向量结构对比")
    print("=" * 80)
    
    if obstacle_nac_info and traffic_light_nac_info:
        print(f"\n{'指标':<30} {'障碍物检测模型':<30} {'红绿灯检测模型':<30} {'差异':<30}")
        print("-" * 120)
        
        obs_nac_length = obstacle_nac_info.get('total_length', 0)
        tl_nac_length = traffic_light_nac_info.get('total_length', 0)
        obs_layer_count = obstacle_nac_info.get('layer_count', 0)
        tl_layer_count = traffic_light_nac_info.get('layer_count', 0)
        
        print(f"{'NAC向量总长度':<30} {obs_nac_length:>30} {tl_nac_length:>30} {obs_nac_length - tl_nac_length:>30}")
        print(f"{'中间层数量':<30} {obs_layer_count:>30} {tl_layer_count:>30} {obs_layer_count - tl_layer_count:>30}")
        
        if obs_nac_length > 0 and tl_nac_length > 0:
            if obs_nac_length == tl_nac_length:
                print("\n✓ 两个模型的NAC向量长度相同，说明它们使用相同的网络结构")
            else:
                print(f"\n⚠ 两个模型的NAC向量长度不同，差异: {abs(obs_nac_length - tl_nac_length)}")
        
        # 对比各层的通道数
        obs_layers = obstacle_nac_info.get('layers_info', [])
        tl_layers = traffic_light_nac_info.get('layers_info', [])
        
        if len(obs_layers) > 0 and len(tl_layers) > 0:
            print("\n各层通道数对比:")
            print(f"{'层序号':<8} {'层名称（简化）':<50} {'障碍物检测':<15} {'红绿灯检测':<15} {'差异':<15}")
            print("-" * 100)
            
            min_len = min(len(obs_layers), len(tl_layers))
            for i in range(min_len):
                obs_info = obs_layers[i]
                tl_info = tl_layers[i]
                layer_short_name = obs_info['name'].split('/')[-2] if '/' in obs_info['name'] else obs_info['name'][:50]
                diff = obs_info['channels'] - tl_info['channels']
                print(f"{obs_info['index']:<8} {layer_short_name[:50]:<50} {obs_info['channels']:<15} {tl_info['channels']:<15} {diff:<15}")
    else:
        print("\n无法获取完整的NAC向量结构信息")
    
    # 所有可用中间层对比
    print("\n" + "=" * 80)
    print("所有可用中间层对比")
    print("=" * 80)
    
    if obstacle_all_layers_info and traffic_light_all_layers_info:
        print(f"\n{'指标':<30} {'障碍物检测模型':<30} {'红绿灯检测模型':<30} {'差异':<30}")
        print("-" * 120)
        
        obs_total_layers = obstacle_all_layers_info.get('total_layers', 0)
        obs_total_channels = obstacle_all_layers_info.get('total_channels', 0)
        obs_conv_count = obstacle_all_layers_info.get('conv_count', 0)
        obs_matmul_count = obstacle_all_layers_info.get('matmul_count', 0)
        
        tl_total_layers = traffic_light_all_layers_info.get('total_layers', 0)
        tl_total_channels = traffic_light_all_layers_info.get('total_channels', 0)
        tl_conv_count = traffic_light_all_layers_info.get('conv_count', 0)
        tl_matmul_count = traffic_light_all_layers_info.get('matmul_count', 0)
        
        print(f"{'总中间层数':<30} {obs_total_layers:>30} {tl_total_layers:>30} {obs_total_layers - tl_total_layers:>30}")
        print(f"{'Conv2D层数':<30} {obs_conv_count:>30} {tl_conv_count:>30} {obs_conv_count - tl_conv_count:>30}")
        print(f"{'MatMul层数':<30} {obs_matmul_count:>30} {tl_matmul_count:>30} {obs_matmul_count - tl_matmul_count:>30}")
        print(f"{'总通道数/特征数':<30} {obs_total_channels:>30,} {tl_total_channels:>30,} {obs_total_channels - tl_total_channels:>30,}")
        
        if obs_total_channels > 0 and tl_total_channels > 0:
            print(f"\n如果使用所有中间层，NAC向量长度将是:")
            print(f"  障碍物检测模型: {obs_total_channels:,}")
            print(f"  红绿灯检测模型: {tl_total_channels:,}")
            print(f"  当前使用的10层: 4,296")
            print(f"  扩展潜力: {obs_total_channels - 4296:,} 个额外通道")
    else:
        print("\n无法获取完整的中间层信息")

if __name__ == "__main__":
    # 设置TensorFlow日志级别
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    main()