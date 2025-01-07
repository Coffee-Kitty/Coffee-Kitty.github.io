
/*

你可以实现一个插件，
专门用于在 Docsify 中自动收集所有文章并按时间顺序显示出来。
以下是实现思路和步骤：



1.自动扫描文档：扫描 Docsify 项目中的所有 Markdown 文件。
2.提取时间信息：从文件名或文章内容的元数据中提取创建/更新时间。
3.生成时间线：根据时间排序生成一份时间线导航页面。
4.动态更新：支持自动更新，无需手动维护。
*/


/*
1. 创建插件框架
Docsify 插件本质是一个 JavaScript 文件，
比如我们是在index.html引用下面这个script来引用docsify的：
  <script src="//cdn.jsdelivr.net/npm/docsify@4"></script>
可以通过挂载在 window.$docsify.plugins 上运行
*/

window.$docsify.plugins = (window.$docsify.plugins || []).concat(function (hook, vm) {
    /*

    5. 渲染到 Docsify 页面
    将生成的时间线内容插入到指定页面（例如 /timeline.md）：
    */

    
    hook.beforeEach(function (content) {
        console.log('Timeline plugin is start.');
        console.log(vm.route.path );
        srcDir='/'
        
        if (vm.route.path === '/') {
            const articles = scanFiles().map((file) => {
                const content = fetch(file).then((res) => res.text());
                    return {
                        path: file,
                        date: extractDate(content),
                        title: extractTitle(content),
                    };
            });
          content = generateTimeline(articles);
        }
        return content; 
      });
  
    hook.ready(function () {
      console.log('Timeline plugin is ready.');
    });
  });

/*

2.扫描文件
可以通过后端服务或 Docsify 的 window.$docsify 配置获取所有 Markdown 文件的路径。
例如，使用 Docsify 的 loadSidebar 配置可以提前加载侧边栏目录：

*/
const scanFiles = () => {
    const sidebarContent = window.$docsify.loadSidebar;
    const files = [];
    console.log(window.$docsify)
    if (sidebarContent) {
        console.log(sidebarContent)

      sidebarContent.split('\n').forEach((line) => {
        const match = /\((.*?)\.md\)/.exec(line);
        if (match) files.push(match[1] + '.md');
      });
    }
    return files;
  };


  /*
 3. 提取时间信息
时间信息可以从文件名或 Markdown 文件的元数据中提取，例如：

文件名格式：2025-01-01-my-article.md
Markdown 文件中的 YAML Front Matter：
---
date: 2025-01-01
title: My Article
---
  
  */

const extractDate = (fileContent) => {
    const match = /date:\s*(\d{4}-\d{2}-\d{2})/.exec(fileContent);
    return match ? match[1] : null;
  };

/*

4. 按时间排序并生成时间线
根据提取到的时间信息排序，生成一个时间线的 Markdown 内容。
*/
const generateTimeline = (articles) => {
    articles.sort((a, b) => new Date(a.date) - new Date(b.date));
    return articles
      .map((article) => `- **${article.date}**: [${article.title}](${article.path})`)
      .join('\n');
  };


/*

6. 使用方式
将插件文件保存为 timeline-plugin.js。
在 index.html 中引入：
html
复制代码
<script src="timeline-plugin.js"></script>
在侧边栏或导航中添加时间线链接：
markdown
复制代码
[时间线](#/timeline)

*/
