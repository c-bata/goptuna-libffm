package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"sync"
	"syscall"

	"github.com/c-bata/goptuna"
	"github.com/c-bata/goptuna/rdb"
	"github.com/c-bata/goptuna/tpe"
	"github.com/jinzhu/gorm"
	"go.uber.org/zap"

	_ "github.com/jinzhu/gorm/dialects/sqlite"
)

func objective(trial goptuna.Trial) (float64, error) {
	lmd, err := trial.SuggestLogUniform("lambda", 1e-6, 1)
	if err != nil {
		return -1, err
	}
	eta, err := trial.SuggestLogUniform("eta", 1e-6, 1)
	if err != nil {
		return -1, err
	}
	latent, err := trial.SuggestInt("latent", 1, 16)
	if err != nil {
		return -1, err
	}
	number, err := trial.Number()
	if err != nil {
		return -1, err
	}
	jsonMetaPath := fmt.Sprintf("./data/optuna/ffm-meta-%d.json", number)

	ctx := trial.GetContext()
	cmd := exec.CommandContext(
		ctx,
		"./ffm-train",
		"-p", "./data/valid2.txt",
		"--auto-stop", "--auto-stop-threshold", "3",
		"-l", fmt.Sprintf("%f", lmd),
		"-r", fmt.Sprintf("%f", eta),
		"-k", fmt.Sprintf("%d", latent),
		"-t", "500",
		"--json-meta", jsonMetaPath,
		"./data/train2.txt",
	)
	stdout := &bytes.Buffer{}
	stderr := &bytes.Buffer{}
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	_ = cmd.Run() // ignore because ffm-train exited with 1 when enabling early stopping.

	var result struct {
		BestIteration int     `json:"best_iteration"`
		BestVALoss    float64 `json:"best_va_loss"`
	}

	jsonStr, err := ioutil.ReadFile(jsonMetaPath)
	if err != nil {
		return -1, fmt.Errorf("failed to read json: %s", err)
	}
	err = json.Unmarshal(jsonStr, &result)
	if err != nil {
		return -1, fmt.Errorf("failed to read json: %s", err)
	}
	if result.BestIteration == 0 && result.BestVALoss == 0 {
		return -1, errors.New("failed to open json meta")
	}

	_ = trial.SetUserAttr("best_iteration", fmt.Sprintf("%d", result.BestIteration))
	_ = trial.SetUserAttr("stdout", stdout.String())
	_ = trial.SetUserAttr("stderr", stderr.String())
	return result.BestVALoss, nil
}

func main() {
	logger, err := zap.NewDevelopment()
	if err != nil {
		os.Exit(1)
	}
	defer logger.Sync()

	// setup storage
	db, err := gorm.Open("sqlite3", "db.sqlite3")
	if err != nil {
		logger.Fatal("failed to open db", zap.Error(err))
	}
	defer db.Close()
	db.DB().SetMaxOpenConns(1)
	storage := rdb.NewStorage(db)

	// create a study
	study, err := goptuna.LoadStudy(
		"goptuna-libffm",
		goptuna.StudyOptionStorage(storage),
		goptuna.StudyOptionSampler(tpe.NewSampler()),
		goptuna.StudyOptionSetLogger(logger),
	)
	if err != nil {
		logger.Fatal("failed to create study", zap.Error(err))
	}

	// create a context with cancel function
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	study.WithContext(ctx)

	// set signal handler
	sigch := make(chan os.Signal, 1)
	defer close(sigch)
	signal.Notify(sigch, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		sig, ok := <-sigch
		if !ok {
			return
		}
		cancel()
		logger.Error("catch a kill signal", zap.String("signal", sig.String()))
	} ()

	// run optimize with context
	concurrency := runtime.NumCPU() - 1
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := study.Optimize(objective, 1000 / concurrency)
			if err != nil {
				logger.Error("optimize catch error", zap.Error(err))
			}
		} ()
	}
	wg.Wait()

	// print best hyper-parameters and the result
	v, _ := study.GetBestValue()
	params, _ := study.GetBestParams()
	logger.Info("result", zap.Float64("value", v),
		zap.String("params", fmt.Sprintf("%#v", params)))
}
