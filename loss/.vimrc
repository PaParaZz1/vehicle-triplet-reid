set nocompatible              " 这是必需的 
filetype off                  " 这是必需的 

" 你在此设置运行时路径 
set rtp+=~/.vim/bundle/Vundle.vim  

" vundle初始化 
call vundle#begin()  

" 这应该始终是第一个 
Plugin 'gmarik/Vundle.vim' 

" 该例子来自https://github.com/gmarik/Vundle.vim README 
Plugin 'tpope/vim-fugitive'  

" 来自http://vim-scripts.org/vim/scripts.html的插件 
Plugin 'L9'  

"未托管在GitHub上的Git插件 
Plugin 'git://git.wincent.com/command-t.git'  


" sparkup vim脚本在名为vim的该软件库子目录下。 
" 传递路径，合理设置运行时路径。 
Plugin 'rstacruz/sparkup', {'rtp': 'vim/'} 

Bundle 'SuperTab'

"每个插件都应该在这一行之前  

call vundle#end()          
set nu 
syntax on
set ts=4
set mouse+=a
set expandtab
set shiftwidth=4
