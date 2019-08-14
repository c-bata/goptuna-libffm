#!/bin/bash

export GO111MODULE=on
go build .
GOOS=linux GOARCH=amd64 go build -o linux-goptuna-libffm .
