<script setup>
import abstract from '../mds/abstract.mdx';
</script>

<template>
  <div>
    <el-divider />

    <el-row justify="center">
        <!-- 1. 在这里添加一个 class，比如 "content-container" -->
        <el-col :xs="24" :sm="20" :md="16" :lg="12" :xl="12" class="content-container">
            <abstract />
        </el-col>
    </el-row>

  </div>
</template>

<style scoped>
/* 2. 在 style 块中为这个 class 添加样式 */
.content-container {
  text-align: justify; /* 这就是实现双边对齐的关键属性 */
  
  /* 为了让单词间距更自然，可以考虑添加下面这个属性，但它不是所有浏览器都支持得很好 */
  /* text-justify: inter-word; */
}
</style>
