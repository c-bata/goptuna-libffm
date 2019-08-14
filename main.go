package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"runtime"

	"github.com/c-bata/goptuna"
	"github.com/c-bata/goptuna/rdb"
	"github.com/c-bata/goptuna/tpe"
	"github.com/jinzhu/gorm"
	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"

	_ "github.com/jinzhu/gorm/dialects/sqlite"
)

func getMetaPath(trialNumber int) string {
	return fmt.Sprintf("./data/optuna/ffm-meta-%d.json", trialNumber)
}

func objective(trial goptuna.Trial) (float64, error) {
	lmd, err := trial.SuggestLogUniform("lambda", 1e-6, 1)
	if err != nil {
		return -1, err
	}
	eta, err := trial.SuggestLogUniform("eta", 1e-6, 1)
	if err != nil {
		return -1, err
	}
	number, err := trial.Number()
	if err != nil {
		return -1, err
	}
	jsonMetaPath := getMetaPath(number)

	libffm := os.Getenv("LIBFFM_CLI")
	if libffm == "" {
		libffm = "./ffm-train"
	}

	cmd := exec.Command(libffm,
		"-p", "./data/valid2.txt",
		"--auto-stop", "--auto-stop-threshold", "3",
		"-l", fmt.Sprintf("%f", lmd),
		"-r", fmt.Sprintf("%f", eta),
		"-k", "4",
		"-t", "500",
		"--json-meta", jsonMetaPath,
		"./data/train2.txt",
	)
	out := &bytes.Buffer{}
	cmd.Stdout = out
	if err = cmd.Run(); err != nil {
		return -1, err
	}

	var result struct {
		BestIteration int     `json:"best_iteration"`
		BestVALoss    float64 `json:"best_va_loss"`
	}

	jsonStr, err := ioutil.ReadFile(jsonMetaPath)
	if err != nil {
		return -1, err
	}
	err = json.Unmarshal(jsonStr, &result)
	if err != nil {
		return -1, err
	}
	if result.BestIteration == 0 && result.BestVALoss == 0 {
		return -1, errors.New("failed to open json meta")
	}

	err = trial.SetUserAttr("best_iteration", fmt.Sprintf("%d", result.BestIteration))
	if err != nil {
		return -1, err
	}
	return result.BestVALoss, nil
}

func main() {
	logger, err := zap.NewDevelopment()
	if err != nil {
		os.Exit(1)
	}
	defer logger.Sync()

	db, err := gorm.Open("sqlite3", "db.sqlite3")
	if err != nil {
		logger.Fatal("failed to open db", zap.Error(err))
	}
	defer db.Close()
	storage := rdb.NewStorage(db)

	study, err := goptuna.LoadStudy(
		"goptuna-libffm",
		goptuna.StudyOptionStorage(storage),
		goptuna.StudyOptionSampler(tpe.NewSampler()),
		goptuna.StudyOptionSetLogger(logger),
	)
	if err != nil {
		logger.Fatal("failed to create study", zap.Error(err))
	}

	eg, ctx := errgroup.WithContext(context.Background())
	study.WithContext(ctx)
	for i := 0; i < runtime.NumCPU()-1; i++ {
		eg.Go(func() error {
			return study.Optimize(objective, 100)
		})
	}
	if err := eg.Wait(); err != nil {
		log.Fatal("got error while optimize")
	}

	v, err := study.GetBestValue()
	if err != nil {
		logger.Fatal("failed to get best value", zap.Error(err))
	}
	params, err := study.GetBestParams()
	if err != nil {
		logger.Fatal("failed to get best params", zap.Error(err))
	}

	fmt.Println("Result:")
	fmt.Println("- best value", v)
	fmt.Println("- best param", params)
}
