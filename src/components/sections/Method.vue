<script lang="ts" setup>
// 在这里设置您想要展示的图片路径
const imagePath = './sgad.png';
</script>

<template>
  <div>
    <el-divider />
    
    <el-row justify="center">
      <h1 class="section-title">Method</h1>
    </el-row>

    <el-row justify="center">
        <!-- 沿用了之前的响应式布局，保证在不同设备上显示效果良好 -->
        <el-col :xs="24" :sm="20" :md="16" :lg="12" :xl="12">
            <!-- 使用标准的 <img> 标签显示图片 -->
            <img :src="imagePath" alt="展示图片" style="width: 100%; height: auto; border-radius: 8px; margin-top: 20px;" />
        </el-col>
    </el-row>

  </div>
</template>

<style scoped>
.section-title {
  font-size: 28px;
  text-align: center;
  margin-bottom: 10px;
}
</style>
