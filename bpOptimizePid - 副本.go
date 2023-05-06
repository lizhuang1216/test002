package main

import (
	"fmt"
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

	Ee0, _ := ctx.GetState("Ee")
	if Ee0 == nil {
		_ = ctx.PutState("Ee", 0)
		Ee0, _ = ctx.GetState("Ee")
	}
	Ee := Ee0.(int)

	err_next0, _ := ctx.GetState("err_next")
	if err_next0 == nil {
		_ = ctx.PutState("err_next", 0)
		err_next0, _ = ctx.GetState("err_next")
	}
	err_next := err_next0.(int)

	udata, Ee, err_next :=  PID_Speed(realspeed, setspeed, Ee, err_next)

	_ = ctx.PutState("Ee", Ee)
	_ = ctx.PutState("err_next", err_next)

	return udata, true
}
//区分聚合函数和通用函数
func (p *bpOptimizedPid) IsAggregate() bool {
	return false
}
//速度PID控制函数
func PID_Speed(realspeed int, setspeed int, Ee int, err_next int) (int, int, int) {
	Kp := 0.22
	Ki := 0.05
	Kd := 0.01
	err := setspeed - realspeed
	Ee = Ee + err
	ires := Ki * float64(Ee)
	if ires > 125 {
		ires = 125
	}
	if ires < -125 {
		ires = -120
	}
	duty_err := int(Kp*float64(err) + ires + Kd*float64(err-err_next))
	if duty_err > 250 {
		duty_err = 250
	}
	if duty_err < 0 {
		duty_err = 0
	}
	err_next = err
	return duty_err, Ee, err_next
}

var BpOptimizedPid  bpOptimizedPid
