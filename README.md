lurenjie's blog.

主题: https://github.com/kitian616/jekyll-TeXt-theme/tree/master

主页链接：https://lu-renjie.github.io/


定义通过 **_粗体加斜体_** 给出。

### 环境配置
```bash
brew install ruby@3.3
# 然后在.zshrc里改PATH加上路径`opt/homebrew/opt/ruby/bin`。可以用`which ruby`检查是是不是brew的ruby。

# 有了ruby后，用以下命令安装Jekyll：
gem install jekyll -v 4.3.3
# 你可能需要换源
# gem sources --add https://gems.ruby-china.com/ --remove https://rubygems.org/
# 在macos上，用brew的ruby安装的，安装路径在opt/homebrew/lib/ruby/gems/3.3.0/bin里，也得改PATH
# 可以通过gem env查看gem会把包安装在哪

bundle install  # 安装一些依赖
```

我的环境版本：
- jekyll 4.3.3
- bundle 4.0.9
- ruby 3.3.11


### 本地运行
```bash
bundle exec jekyll serve
```
