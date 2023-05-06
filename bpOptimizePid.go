package main

import (
	"fmt"
	"math/rand"
	"math"
	"github.com/lf-edge/ekuiper/pkg/api"
)

type bpOptimizedPid struct {     //定义pidSpeed结构体
}
//Validate校验传入的参数是否正确
func (p *bpOptimizedPid) Validate(args []interface{}) error{
	if len(args) != 2{
		return fmt.Errorf("echo function only supports 2 parameter but got %d", len(args))
	}
	return nil
}
//主程序执行
func (p *bpOptimizedPid) Exec(args []interface{}, ctx api.FunctionContext) (interface{}, bool) {

	//realspeed := int(args[0].(float64))
	//setspeed := int(args[1].(float64))
	realspeed, _ := args[0].(int)
	setspeed, _ := args[1].(int)

	xite := 0.25   // 学习因子
	alfa := 0.05   // 惯量因子
	S := 1         // Signal type

	// NN Structure
	IN := 4   // 输入层个数
	H := 5    // 隐藏层个数
	Out := 3  // 输出层个数

	wi := [][]float64{{-0.6394, -0.2696, -0.3756, -0.7023},
		{-0.8603, -0.2013, -0.5024, -0.2596},
		{-1.0749, 0.5543, -1.6820, -0.5437},
		{-0.3625, -0.0724, -0.6463, -0.2859},
		{0.1425, 0.0279, -0.5406, -0.7660}}
	// wi = 0.50 * rand.Float64() * H * IN
	wi_1 := wi
	wi_2 := wi
	wi_3 := wi

	wo := [][]float64{{0.7576, 0.2616, 0.5820, -0.1416, -0.1325},
		{-0.1146, 0.2949, 0.8352, 0.2205, 0.4508},
		{0.7201, 0.4566, 0.7672, 0.4962, 0.3632}}
	// wo = 0.50 * rand.Float64() * Out * H
	wo_1 := wo
	wo_2 := wo
	wo_3 := wo

	x := []float64{0, 0, 0}
	u_1 := 0.0
	u_2 := 0.0
	u_3 := 0.0
	u_4 := 0.0
	u_5 := 0.0
	y_1 := 0.0
	y_2 := 0.0
	y_3 := 0.0

	// 初始化
	Oh := make([]float64, H)   // 从隐藏层到输出层
	I := Oh                   // 从输入层到隐藏层
	error_2 := 0.0
	error_1 := 0.0
	




	_ = ctx.PutState("Ee", Ee)
	_ = ctx.PutState("err_next", err_next)

	return udata, true
}
//区分聚合函数和通用函数
func (p *bpOptimizedPid) IsAggregate() bool {
	return false
}
//速度PID控制函数
func BP_PID(realspeed int, setspeed int, Ee int, err_next int) (int, int, int) {
			time := float64(k) * ts

		rin := 1.0

		// 非线性模型
		a := 1.2 * (1 - 0.8 * math.Exp(-0.1 * float64(k)))
		yout := a * y_1 / (1 + math.Pow(y_1, 2)) + u_1  // 输出

		error := rin - yout  // 误差
		xi := []float64{rin, yout, error, 1}

		x[0] = error - error_1
		x[1] = error
		x[2] = error - 2 * error_1 + error_2

		epid := []float64{x[0], x[1], x[2]}
		I = matMul(xi, transpose(wi))
		for j := 0; j < H; j++ {
			Oh[j] = (math.Exp(I[j]) - math.Exp(-I[j])) / (math.Exp(I[j]) + math.Exp(-I[j]))  // Middle Layer
		}
		K := matMul(wo, Oh)  // Output Layer
		for l := 0; l < Out; l++ {
			K[l] = math.Exp(K[l]) / (math.Exp(K[l]) + math.Exp(-K[l]))  // Getting kp, ki, kd
		}
		kp := K[0]
		ki := K[1]
		kd := K[2]
		Kpid := []float64{kp, ki, kd}

		du := dot(Kpid, epid)
		u := u_1 + du


		// 饱和限制
		if u >= 10 {
			u = 10
		}
		if u <= -10 {
			u = -10
		}
		if math.Abs(u-u_1) > 0.2 {
			if u > u_1 {
				u = u_1 + 0.2
			} else {
				u = u_1 - 0.2
			}
		}

		duy := (u - 2*u_1 + u_2) / math.Pow(ts, 2)
		dy := (y_1 - y_2) / ts

		// 更新权值
		epo := []float64{x[0], x[1], x[2], duy, dy}
		tempwo := wo
		tempwi := wi
		wo = matAdd(wo, matScalar(matMul(transpose(Kpid), epo), xite))  // Update output weights
		wi = matAdd(wi, matScalar(matMul(transpose(epid), matMul(tempwo, transpose(Oh))), xite))  // Update input weights

		// 更新偏置
		tempwo_1d := make([]float64, Out)
		for i := 0; i < Out; i++ {
			tempwo_1d[i] = tempwo[i][0]
		}
		tempwo_2d := [][]float64{tempwo_1d}
		wo_1d := make([]float64, Out)
		for i := 0; i < Out; i++ {
			wo_1d[i] = wo[i][0]
		}
		wo_2d := [][]float64{wo_1d}
		wo_1 = matAdd(wo_2d, matScalar(matMul(xite, tempwo_2d), transpose(Kpid)))  // Update bias weights of output layer

		tempwi_1d := make([]float64, H)
		for i := 0; i < H; i++ {
			tempwi_1d[i] = tempwi[i][0]
		}
		tempwi_2d := [][]float64{tempwi_1d}
		wi_1d := make([]float64, H)
		for i := 0; i < H; i++ {
			wi_1d[i] = wi[i][0]
		}
		wi_2d := [][]float64{wi_1d}
		wi_1 = matAdd(wi_2d, matScalar(matMul(xite, tempwi_2d), transpose(epid)))  // Update bias weights of input layer

		// 更新状态变量
		error_2 = error_1
		error_1 = error
		u_5 = u_4
		u_4 = u_3
		u_3 = u_2
		u_2 = u_1
		u_1 = u
		y_3 = y_2
		y_2 = y_1
		y_1 = yout
		wi_3 = wi_2
		wi_2 = wi_1
		wo_3 = wo_2
		wo_2 = wo_1
	return duty_err, Ee, err_next
}

var BpOptimizedPid  bpOptimizedPid
