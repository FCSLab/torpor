#!/bin/bash
set -x

cd ../proto
protoc -I=. --python_out=. signal.proto
cd -

mv ../proto/signal_pb2.py .