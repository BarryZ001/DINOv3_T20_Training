#!/usr/bin/env python3
"""
燧原T20 8卡分布式连接测试脚本
用于验证分布式环境是否正常工作
基于燧原官方最佳实践
"""

import os
import sys
import torch

# 🔧 设置燧原GCU环境变量 - 基于官方最佳实践
os.environ.setdefault('ENFLAME_CLUSTER_PARALLEL', 'true')
os.environ.setdefault('ENFLAME_ENABLE_EFP', 'true')
os.environ.setdefault('TOPS_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
os.environ.setdefault('OMP_NUM_THREADS', '5')
os.environ.setdefault('ECCL_ASYNC_DISABLE', 'false')
os.environ.setdefault('ENABLE_RDMA', 'true')
os.environ.setdefault('ECCL_MAX_NCHANNELS', '2')
os.environ.setdefault('ENFLAME_UMD_FLAGS', 'mem_alloc_retry_times=1')
os.environ.setdefault('ECCL_RUNTIME_3_0_ENABLE', 'true')
os.environ.setdefault('ENFLAME_PT_EVALUATE_TENSOR_NEEDED', 'false')

# 禁用CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    import torch_gcu
    torch_gcu_available = True
except ImportError:
    torch_gcu_available = False

def test_distributed():
    """测试分布式连接"""
    
    print("🔧 测试燧原T20 8卡分布式连接...")
    
    # 获取分布式参数
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    print(f"🔧 分布式参数: local_rank={local_rank}, world_size={world_size}, rank={rank}")
    
    # 检查GCU设备
    if torch_gcu_available:
        try:
            device_count = torch_gcu.device_count()
            print(f"🔍 检测到 {device_count} 个 GCU 设备")
            
            # 设置当前设备
            torch_gcu.set_device(local_rank)
            current_device = torch_gcu.current_device()
            print(f"🔧 当前进程使用 GCU 设备: {current_device}")
            
            # 创建测试张量
            test_tensor = torch.randn(4, 4).to(f'gcu:{current_device}')
            print(f"✅ 成功在 GCU:{current_device} 上创建张量: {test_tensor.shape}")
            
        except Exception as e:
            print(f"⚠️ GCU 操作失败: {e}")
            return False
    else:
        print("⚠️ torch_gcu 不可用，使用 CPU")
    
    # 测试分布式通信
    if world_size > 1:
        try:
            print("🔧 初始化分布式进程组...")
            torch.distributed.init_process_group(
                backend='eccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            print("✅ 分布式进程组初始化成功")
            
            # 创建测试张量进行all_reduce
            if torch_gcu_available:
                test_tensor = torch.ones(2, 2).to(f'gcu:{local_rank}') * rank
            else:
                test_tensor = torch.ones(2, 2) * rank
            
            print(f"Rank {rank} 原始张量: {test_tensor}")
            
            # 执行all_reduce
            torch.distributed.all_reduce(test_tensor)
            print(f"Rank {rank} all_reduce后张量: {test_tensor}")
            
            torch.distributed.barrier()
            print(f"✅ Rank {rank} 分布式通信测试成功")
            
        except Exception as e:
            print(f"❌ Rank {rank} 分布式通信失败: {e}")
            return False
    else:
        print("🔧 单卡模式，跳过分布式通信测试")
    
    print(f"✅ Rank {rank} 所有测试通过!")
    return True

if __name__ == '__main__':
    success = test_distributed()
    sys.exit(0 if success else 1)
