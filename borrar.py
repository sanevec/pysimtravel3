
	def aStartV2(self,cell):
		# only mark visited if it has more than one destination
		visited={}
		visited[cell]=0
		opened={}
		for d in cell.destination:
			if d.car==None:
				opened[d]=(d,0)
			else:
				opened[d]=(d,1)
		opened2={}
		while True:
			# ordena los opened por d (densidad) ascendente
			openedLista=sorted(opened.items(), key=lambda x: x[1][1])

			for (o,rd) in openedLista:
				if o.x==self.targetx and o.y==self.targety:
					return opened[o][0]
				r=rd[0] # response
				d=rd[1] # density
				if len(o.destination)==1:
					destinantion=o.destination[0]
					if destinantion.car==None:
						opened2[destinantion]=(r,d)
					else:	
						opened2[destinantion]=(r,d+1)
				else:
					g=visited.get(o)
					if g==None or g>d:
						visited[o]=d
						for des in o.destination:
							if des.car==None:
								opened2[des]=(r,d)
							else:
								opened2[des]=(r,d+1)
			opened=opened2
			opened2={}

