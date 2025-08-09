<script lang="ts" setup>
// 在这里设置您想要展示的图片路径
const imagePath = './teaser.png';
</script>

<template>
  <div>
    <el-divider />

    <el-row justify="center">
        <!-- 沿用了之前的响应式布局，保证在不同设备上显示效果良好 -->
        <el-col :xs="24" :sm="20" :md="16" :lg="12" :xl="12">
            <!-- 使用标准的 <img> 标签显示图片 -->
            <img :src="imagePath" style="width: 100%; object-fit: contain;">
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
