#!/usr/bin/env python3
"""
GCU驱动修复脚本
用于解决燧原T20 GCU驱动层面的Admin Queue RAS和topsStreamCreate错误
"""

import os
import sys
import subprocess
import time

def run_command(cmd, description):
    """执行系统命令并返回结果"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ {description} 成功")
            if result.stdout.strip():
                print(f"   输出: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} 失败")
            if result.stderr.strip():
                print(f"   错误: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 超时")
        return False
    except Exception as e:
        print(f"❌ {description} 异常: {e}")
        return False

def main():
    print("🚨 GCU驱动修复脚本")
    print("=" * 50)
    
    # 1. 检查GCU设备状态
    print("\n📊 检查GCU设备状态...")
    run_command("lspci | grep -i enflame", "检查GCU硬件")
    run_command("ls -la /dev/dtu*", "检查GCU设备节点")
    
    # 2. 检查驱动模块
    print("\n🔍 检查驱动模块...")
    run_command("lsmod | grep -E '(dtu|gcu|enflame)'", "检查已加载的驱动模块")
    
    # 3. 检查系统资源
    print("\n💾 检查系统资源...")
    run_command("free -h", "检查内存使用")
    run_command("df -h", "检查磁盘空间")
    
    # 4. 尝试重置GCU设备
    print("\n🔄 尝试重置GCU设备...")
    
    # 设置严格的环境变量
    reset_env = {
        'GCU_RESET_ON_ERROR': '1',
        'GCU_FORCE_RESET': '1',
        'TOPS_RESET_DEVICE': '1',
        'DTU_RESET_ON_INIT': '1',
        'GCU_CLEAR_CACHE': '1',
        'GCU_REINIT_DRIVER': '1'
    }
    
    for key, value in reset_env.items():
        os.environ[key] = value
        print(f"   设置 {key}={value}")
    
    # 5. 尝试重新加载驱动（需要root权限）
    print("\n🔧 尝试重新初始化驱动...")
    
    # 检查是否有root权限
    if os.geteuid() == 0:
        print("✅ 检测到root权限，尝试重新加载驱动...")
        
        # 卸载相关模块
        modules_to_unload = ['torch_gcu', 'dtu_drv', 'efdrv']
        for module in modules_to_unload:
            run_command(f"rmmod {module}", f"卸载模块 {module}")
            time.sleep(1)
        
        # 重新加载驱动
        run_command("modprobe efdrv", "重新加载efdrv驱动")
        time.sleep(2)
        
        # 重置设备权限
        run_command("chmod 666 /dev/dtu*", "重置设备权限")
        
    else:
        print("⚠️  没有root权限，无法重新加载驱动")
        print("   建议以root用户运行此脚本或使用sudo")
    
    # 6. 创建最小化测试
    print("\n🧪 创建最小化GCU测试...")
    
    minimal_test = '''
import os
import sys

# 设置最保守的环境变量
os.environ.update({
    'GCU_DISABLE_AMP': '1',
    'GCU_FORCE_FP32': '1', 
    'GCU_MEMORY_FRACTION': '0.5',
    'GCU_SYNC_ALLOC': '1',
    'GCU_DISABLE_CACHING': '1',
    'GCU_SINGLE_DEVICE': '1',
    'PYTORCH_GCU_ALLOC_CONF': 'max_split_size_mb:32,garbage_collection_threshold:0.3',
    'TOPS_VISIBLE_DEVICES': '0'  # 只使用第一个设备
})

try:
    import torch
    torch.set_default_dtype(torch.float32)
    print("✅ PyTorch导入成功")
    
    import torch_gcu
    print("✅ torch_gcu导入成功")
    
    if torch_gcu.is_available():
        print(f"✅ 检测到 {torch_gcu.device_count()} 个GCU设备")
        
        # 只测试第一个设备
        device = 'xla:0'
        print(f"🧪 测试设备: {device}")
        
        # 最小张量测试
        x = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
        print(f"✅ 最小张量创建成功: {x}")
        
        # 清理
        del x
        torch_gcu.empty_cache()
        print("✅ 内存清理成功")
        
    else:
        print("❌ GCU设备不可用")
        
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
'''
    
    with open('/tmp/minimal_gcu_test.py', 'w') as f:
        f.write(minimal_test)
    
    print("📝 最小化测试脚本已创建: /tmp/minimal_gcu_test.py")
    
    # 7. 运行最小化测试
    print("\n🚀 运行最小化测试...")
    success = run_command("cd /tmp && python3 minimal_gcu_test.py", "最小化GCU测试")
    
    # 8. 总结和建议
    print("\n" + "=" * 50)
    print("📋 修复总结和建议:")
    
    if success:
        print("✅ 最小化测试通过，GCU驱动可能已修复")
        print("💡 建议:")
        print("   1. 重新运行完整的训练脚本")
        print("   2. 如果仍有问题，考虑重启服务器")
    else:
        print("❌ 最小化测试失败，需要进一步处理")
        print("💡 建议:")
        print("   1. 重启服务器以完全重置GCU驱动状态")
        print("   2. 检查燧原T20驱动版本是否与PyTorch GCU版本兼容")
        print("   3. 联系燧原技术支持，提供Admin Queue RAS错误日志")
        print("   4. 考虑降级到更稳定的驱动版本")
    
    print("\n🔧 环境变量建议:")
    print("   export GCU_RESET_ON_ERROR=1")
    print("   export GCU_SINGLE_DEVICE=1") 
    print("   export TOPS_VISIBLE_DEVICES=0")
    print("   export GCU_MEMORY_FRACTION=0.5")

if __name__ == '__main__':
    main()