# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:43:31 2019

@author: zxy
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MyTools import reverseDict



if __name__ == "__main__":
	plt.style.use("ggplot")#设置图形风格
	
	print("图形风格:",plt.style.available)#图形风格
	x = np.linspace(0,10,100)
	data = np.random.randn(2, 100)
	fig1 = plt.figure()
	plt.plot(x, np.sin(x))
	plt.plot(x,np.cos(x))
	fig1.show()
	fig1.savefig("paintTestFile/1.eps")

	supportedTypes1 = fig1.canvas.get_supported_filetypes()
	print("打印支持的存储格式:",supportedTypes1)
	np.random.seed(111111)
	data = np.random.randn(2, 100)
	
	
	fig2, axs = plt.subplots(2, 2, figsize=(5, 5))
	
	
	axs[0, 0].hist(data[0])
	axs[1, 0].scatter(data[0], data[1])
	axs[0, 1].plot(data[0], data[1])
	axs[1, 1].hist2d(data[0], data[1])
	axs[1, 1].hist2d(x, np.sin(x))
	

	
	print("matplotlib支持的颜色：")
				#	=====		=======
				#	Alias		Color
				#	=====		=======
				#	'b'		blue
				#	'g'		green
				#	'r'		red
				#	'c'		cyan
				#	'm'		magenta
				#	'y'		yellow
				#	'k'		black
				#	'w'		white
				#	=====		=======
				
	#1、一般情况下首先绘制一个fig：容纳各种坐标轴，图形，文字和标签的容器，即图形画布
	fig3 = plt.figure()
	#绘制一个带有刻度和标签的矩形
	ax = plt.axes()#add an axes to figure
	#使用ax绘图
	x = np.linspace(-20,20,1000)
	ax.plot(x,np.arcsinh(x))#同一个矩形中绘制多个图形
	ax.plot(x,np.arcsinh(x+2))#同一个矩形中绘制多个图形
	#2、调整图形：线条的颜色与风格
	ax.plot(x,x,linestyle="--",color="y")#黄色
	ax.plot(x,np.sin(x+1),linestyle="-.",color="k")#黑色
	ax.plot(x,np.sin(x+2),linestyle="-.",color="r")#红色
	ax.plot(x,np.sin(x+3),linestyle=":",color="g")#绿色
	#3、调整图形：坐标轴上下限
	ax.set_xlim(0,10)#正序
	ax.set_ylim(2,0)#逆序
	ax.axis([0,10,10,0])#一行代码设置
	ax.axis("tight")#使图形变紧凑，去除空白边缘
	#4、设置坐标平面的背景颜色
	ax.set_facecolor((0.5,0.6,0.1))#rgb的颜色比例，也可直接设颜色字符
	#5、设置图形标签
		#5.1:图形标题
	ax.set_title("I'm awesome!")
		#5.2:坐标轴标题
	ax.set_xlabel("x's value")
	ax.set_ylabel("y's value")
		#5.3:显示图例
	ax.legend(["a","b","c","d"])#可在ax.plot中设label，也可在此处按顺序定义图例
	#6、一次性设置所有属性
	ax.set(xlim=[-10,10],ylim=[-5,5],title="True awesome",xlabel="x",ylabel="sin(x)")
	
	
	
	
	#1、绘制散点图
	fig4 = plt.figure()
	ax1 = plt.axes()
	x = np.linspace(-10,10,100)
	y = np.sin(x)
	ax1.plot(x,y,linestyle='--',marker="o",markersize=3,color="y")
	#2、simple example
	fig5 = plt.figure()
	ax2 = plt.axes()
	markers = ['o','.',',','x','+','v','<','>','s','d']
	rng = np.random.RandomState(0)
	for marker in markers:
		ax2.plot(rng.rand(5),rng.rand(5),marker ,label="marker='{}'".format(marker))#每次绘制5个点
		ax2.plot(rng.rand(5),rng.rand(5),linestyle='-.')
	ax2.legend(numpoints=1)
	ax2.set_xlim(0,1.8)
	#3、scatter绘制散点图
	fig6 = plt.figure()
	ax3 = plt.axes()
	ax3.scatter(x,y,marker="o",color='c')
	#4、自定义散点图
	fig7 = plt.figure()
	fig7.set_size_inches(8,6)#设置画布大小
	ax4 = plt.axes()
	rng = np.random.RandomState(100)
	x = rng.randn(100)#正态分布的数据
	y = rng.randn(100)
	colors = rng.randn(100)
	size = rng.randn(100)*100#size以像素点为单位
	ax4.scatter(x,y,c=colors,s=size,alpha=0.4,cmap="viridis_r")#cmap对应了一张颜色表，将颜色c的数值映射成具体的色彩
	
	
	
	
	
	#可视化异常处理
	#1、基本误差线
	fig8 = plt.figure()
	fig8.set_size_inches(8,6)
	ax5 = plt.axes()
	x = np.linspace(0,10,10)
	dx = 0.5#x轴的正负误差值
	dy = 20#y轴的正负误差值
	y = np.sin(x) + dy * np.random.randn(10)
	ax5.errorbar(x,y,xerr=dx,yerr=dy,fmt="ok",ecolor="gray",elinewidth=3,capsize=10)
	
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	plt.show()