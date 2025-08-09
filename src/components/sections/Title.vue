<script lang="ts" setup>

import { ElIcon } from 'element-plus'
import { Document, Files, MagicStick, Picture, DataAnalysis, Film, Message } from '@element-plus/icons-vue'
import GithubIcon from '@/components/sections/GithubIcon.vue'
import ArxivIcon from '@/components/sections/ArxivIcon.vue'
import PdfIcon from '@/components/sections/PdfIcon.vue'

// logo地址，没有则置为""即可
const logo = ''

// 标题
const title = 'SGAD: Semantic and Geometric-aware Descriptor for Local Feature Matching'

// 标题颜色
const title_color = '#000000'

// 标题补充，没有则置为''即可
const title_supp = ''

// 标题补充颜色
const title_supp_color = '#42B883'

// 按钮颜色
const btn_color = '#444444'

// 作者清单（包含作者姓名、头像、主页、地址序号）
const authors = [
  {
    name: "Xiangzeng Liu",
    icon: "",
    homepage: "",
    address_flag: "1*"
  },
  {
    name: "Chi Wang",
    icon: "",
    homepage: "https://github.com/mr-chiwang",
    address_flag: "1*†"
  },
  {
    name: "Guanglu Shi",
    icon: "",
    homepage: "",
    address_flag: "1"
  },
  {
    name: "Xiaodong Zhang",
    icon: "",
    homepage: "",
    address_flag: "1"
  },
  {
    name: "Qiguang Miao",
    icon: "",
    homepage: "",
    address_flag: "1†"
  },
  {
    name: "Miao Fan",
    icon: "",
    homepage: "",
    address_flag: "2"
  },
]

// 地址清单（包含地址名称、头像、主页、地址序号）
const addresses = [
  {
    address_flag: "1",
    name: "Xidian University"
  },
  {
    address_flag: "2",
    name: "Navinfo Europe B.V"
  },
]

// 共一和通讯提示
const con_and_corresponding_author =
  "*Equal Contribution.   †Corresponding Author."

// 邮箱地址
const emails = [
  "xzliu@xidian.edu.cn",
  "chiwang@stu.xidian.edu.cn"
]

// 强调内容
const emphases = [
  "ICCV 2025 Highlight"
]

// 提供引导资料链接
const buttons = [
  {
    disabled: false,
    name: "Paper",
    link: "https://arxiv.org/pdf/2508.02278",
    component: PdfIcon,
  },
  {
    disabled: false,
    name: "arXiv",
    link: "http://arxiv.org/abs/2508.02278",
    component: ArxivIcon,
  },
  {
    disabled: false,
    name: "Code",
    link: "https://github.com/mr-chiwang/SGAD",
    component: GithubIcon,
  },
  {
    disabled: true,
    name: "Poster",
    component: Picture,
  },
// {
//   disabled: false,
//   name: "Video",
//   link: "",
//   component: Film,
// },
]

</script>

<template>
  <div>
    <!-- 文章logo -->
    <el-row v-if="logo" justify="center">
      <el-image :src="logo" class="logo" fit="cover" />
    </el-row>

    <!-- 文章标题 -->
    <el-row justify="center">
      <el-col :span="20">
        <h1 class="paper-title">
          <span v-if="title" :style="{color:title_color}"> {{ title }}</span>
          <span v-if="title_supp" :style="{color:title_supp_color}"> {{ title_supp }}</span>
        </h1>
      </el-col>
    </el-row>

    <!-- 强调内容 -->
    <el-row justify="center" class="emphasis" v-for="emphasis in emphases">
        {{ emphasis }}
    </el-row>

  <!-- 作者名单 -->
  <el-row justify="center">
    <!-- 
      我们使用 <template> 标签进行循环，这样可以不对 DOM 产生额外影响。
      v-if/v-else 指令必须是相邻的兄弟元素。
    -->
    <template v-for="author in authors" :key="author.name">
      
      <!-- 情况一: 如果 author.homepage 存在 (不为空字符串), 则渲染成一个带链接的蓝色按钮 -->
      <a v-if="author.homepage" :href="author.homepage">
        <el-button class="title-button" type="primary" text>
          <el-avatar v-if="author.icon" :size="40" :src="author.icon" />
          <span class="author">
            {{ author.name }}<sup v-if="author.address_flag" class="name_sup">{{ author.address_flag }}</sup>
          </span>
        </el-button>
      </a>

      <!-- 情况二: 否则 (author.homepage 为空), 则渲染成一个普通的、不可点击的文字按钮 -->
      <el-button v-else class="title-button" text>
        <el-avatar v-if="author.icon" :size="40" :src="author.icon" />
        <span class="author">
          {{ author.name }}<sup v-if="author.address_flag" class="name_sup">{{ author.address_flag }}</sup>
        </span>
      </el-button>

    </template>
  </el-row>


    <!-- 地址名单 -->
    <el-row justify="center">
      <el-button class="title-button" text v-for="address in addresses" :key="address.address_flag">        <el-avatar v-if="address.icon" :size="40" :src="address.icon" />
        <span class="address">
          <sup v-if="address.address_flag" class="address_sup">{{ address.address_flag }}</sup>{{ address.name }}
        </span>
      </el-button>
    </el-row>


    <!-- 共一和通讯提示内容 -->
    <el-row justify="center" class="con-cor">
        {{ con_and_corresponding_author }}
    </el-row>

    <!-- 邮箱 -->
    <el-row justify="center" class="email-row">
        <!-- 邮箱图标 -->
        <el-icon :size="16" color="#606266" style="margin-right: 8px;"><Message /></el-icon>
        <!-- 循环渲染邮箱地址 -->
        <template v-for="(email, index) in emails" :key="email">
            <a :href="'mailto:' + email" class="email-link">{{ email }}</a>
            <span v-if="index < emails.length - 1" class="email-separator">,</span>
        </template>
    </el-row>

    <!-- 提供引导按钮 -->
    <el-row justify="center" style="margin-bottom: 20px;">
      <el-col :span="20">
        <el-row justify="center">
          <a :href=button.link v-for="button in buttons">
            <el-button class="guidance-button" size="default" :color="btn_color" :disabled="button.disabled" round>
              <el-icon :size="18">
                <component :is="button.component" />
              </el-icon>
              <span class="btn-text">{{ button.name }}</span>
            </el-button>
          </a>
        </el-row>
      </el-col>
    </el-row>

  </div>
</template>

<style scoped>

/* 文章标题字体、字间距、居中排布、字号 */
.paper-title {
  font-family: "MyFont", Verdana, sans-serif;
  letter-spacing: 2px;
  font-size: 42px;
  margin: 32px;
  text-align: center;
}

/* 姓名和地址按钮 */
.title-button {
  margin: 10px 3px;
}

/* 姓名和地址按钮光标悬浮 */
.title-button:hover {
  margin: 10px 8px;
}

/* 引导材料按钮 */
.guidance-button {
  margin: 8px 5px;
  box-shadow: #d8d8d8 1px 1px 1px 1px;
}

/* 姓名属性 */
.author {
  font-size: 18px;
  margin-left: 3px;
}

/* 姓名上标属性 */
.name_sup {
  color: #606266;
  margin-left: 3px;
}

/* 地址属性 */
.address {
  font-size: 18px;
}

/* 地址上标属性 */
.address_sup {
  color: #606266;
  margin-right: 1px;
}

/* 头像属性 */
.el-avatar {
  margin-right: 6px;
  box-shadow: #b7b7b7 0px 0px 3px 1px;
}

/* 共一和通讯文字属性 */
.con-cor {
  font-family: Arial;
  font-size: 14px;
  margin: 18px 0px;
  text-align: center;
}

/* 强调信息属性 */
.emphasis {
  color: chocolate;
  font-weight: bold;
  margin: 8px;
  font-size: 22px;
  text-align: center;
}

/* 引导材料按钮文字属性 */
.btn-text {
  font-size: 18px;
  color: #ffffff;
}

.el-alert {
  margin: 10px 0 0;
}

.el-alert:first-child {
  margin: 0;
}

.logo {
  width: 150px;
  height: 150px;
  border-radius: 50%;
  box-shadow: #ced3dc 0px 0px 3px 2px;
  margin-top: 40px;
}

/* 手机端链接样式处理 */
a:-webkit-any-link {
  text-decoration: none;
}

/* 取消鼠标焦点悬浮在链接上的颜色装饰 */
a:hover {
  color: inherit;
  border-bottom: none;
}

/* 链接装饰，取消下划线和链接颜色 */
a {
	text-decoration: None;
	color: inherit;
}

/* 邮箱区域样式 */
.email-row {
  display: flex;
  align-items: center;
  margin-top: -10px; /* 微调与上方元素的间距 */
  margin-bottom: 25px; /* 增加与下方按钮的间距 */
}

/* 邮箱链接样式 */
.email-link {
  font-family: 'Courier New', Courier, monospace;
  font-size: 16px;
  color: #337ab7; /* 标准链接蓝色 */
  text-decoration: none;
}
.email-link:hover {
  text-decoration: underline;
}

/* 邮箱分隔符样式 */
.email-separator {
  margin: 0 8px;
  color: #606266;
}

</style>
