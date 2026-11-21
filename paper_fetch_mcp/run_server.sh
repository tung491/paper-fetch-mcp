#!/usr/bin/env bash
# Run PaperFetchServer as a background HTTP service on port 8766.
# Usage:
#   ./run_server.sh start   - Start the server
#   ./run_server.sh stop    - Stop the server
#   ./run_server.sh restart - Restart the server
#   ./run_server.sh status  - Check if running
#   ./run_server.sh logs    - Tail the log file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$SCRIPT_DIR/.server.pid"
LOG_FILE="$SCRIPT_DIR/server.log"
HOST="0.0.0.0"
PORT="8766"
TRANSPORT="streamable-http"
HEALTH_URL="http://127.0.0.1:$PORT/health"
HEALTH_TIMEOUT_SEC=60
HEALTH_POLL_INTERVAL=0.5

# Poll $HEALTH_URL until it answers, the PID dies, or HEALTH_TIMEOUT_SEC elapses.
# Returns 0 (success) as soon as curl gets ANY HTTP response -- a real 200 is
# printed as the health JSON; a 404 (the /health route not deployed yet on the
# running process -- see telemetry.py rollout) or any other HTTP status is
# still treated as success, just with a warning, because it proves the server
# process is up and answering requests. Only "no response at all" (connection
# refused, timeout) counts as "not up yet" and keeps the loop going. Returns 1
# only if the deadline passes or the PID exits first, printing the log tail.
wait_for_health() {
    local pid="$1"
    local deadline=$((SECONDS + HEALTH_TIMEOUT_SEC))
    local body_file http_code curl_exit
    body_file="$(mktemp)"

    while true; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Server process (PID $pid) exited before answering $HEALTH_URL."
            echo "Last lines of $LOG_FILE:"
            tail -n 20 "$LOG_FILE" 2>/dev/null
            rm -f "$body_file"
            return 1
        fi

        set +e
        http_code=$(curl -sf -m 2 -o "$body_file" -w '%{http_code}' "$HEALTH_URL" 2>/dev/null)
        curl_exit=$?
        set -e

        if [ "$curl_exit" -eq 0 ]; then
            echo "Server healthy (HTTP $http_code):"
            cat "$body_file"
            echo
            rm -f "$body_file"
            return 0
        fi

        if [ -n "$http_code" ] && [ "$http_code" != "000" ]; then
            if [ "$http_code" = "404" ]; then
                echo "WARNING: $HEALTH_URL returned 404 (route not deployed on the running process yet, see telemetry.py rollout). Server process is up; treating startup as successful."
            else
                echo "WARNING: $HEALTH_URL returned unexpected HTTP $http_code. Server process is up; treating startup as successful."
            fi
            rm -f "$body_file"
            return 0
        fi

        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "Server did not respond on $HEALTH_URL within ${HEALTH_TIMEOUT_SEC}s."
            echo "Last lines of $LOG_FILE:"
            tail -n 20 "$LOG_FILE" 2>/dev/null
            rm -f "$body_file"
            return 1
        fi

        sleep "$HEALTH_POLL_INTERVAL"
    done
}

start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Server already running (PID $(cat "$PID_FILE"))"
        return 1
    fi

    export PYTHONUNBUFFERED=1
    cd "$PROJECT_DIR"
    PYTHONUNBUFFERED=1 setsid "$PROJECT_DIR/.venv/bin/python" -m paper_fetch_mcp \
        --transport "$TRANSPORT" \
        --host "$HOST" \
        --port "$PORT" \
        > "$LOG_FILE" 2>&1 < /dev/null &

    PID=$!
    echo $PID > "$PID_FILE"
    disown $PID

    if wait_for_health "$PID"; then
        echo "Server started (PID $(cat "$PID_FILE"))"
        echo "URL: http://$HOST:$PORT/mcp"
    else
        echo "Server failed to start. Check $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No PID file found. Server not running?"
        return 0
    fi

    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping server (PID $PID)..."
        kill "$PID"
        sleep 2
        if kill -0 "$PID" 2>/dev/null; then
            echo "Force killing..."
            kill -9 "$PID"
        fi
        echo "Server stopped."
    else
        echo "Process $PID not running."
    fi
    rm -f "$PID_FILE"
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Server running (PID $(cat "$PID_FILE"))"
        echo "URL: http://$HOST:$PORT/mcp"
    else
        echo "Server not running."
        [ -f "$PID_FILE" ] && rm -f "$PID_FILE"
    fi
}

logs() {
    tail -f "$LOG_FILE"
}

case "${1:-}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; sleep 1; start ;;
    status)  status ;;
    logs)    logs ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
