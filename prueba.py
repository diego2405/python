def incrementar(i):
	i[0]+=1

def llamar():
	i = [5]
	print 'antes',i[0]
	incrementar(i)
	print 'despues',i[0]

if __name__ == '__main__':
	llamar()
	linea = ''
	linea += 'hola'
	linea += '%f' % (5+6.3)
	p = 'erroor'
	if p != 'error':
		print 'distinto'
	print linea