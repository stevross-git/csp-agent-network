#!/bin/bash
# Helper script to manage the CSP infrastructure

case "$1" in
    start)
        echo "Starting all services..."
        docker-compose -f docker-compose.databases.yml up -d
        docker-compose -f monitoring/docker-compose.monitoring.yml up -d
        docker-compose -f monitoring/docker-compose.exporters.yml up -d
        ;;
    stop)
        echo "Stopping all services..."
        docker-compose -f monitoring/docker-compose.exporters.yml down
        docker-compose -f monitoring/docker-compose.monitoring.yml down
        docker-compose -f docker-compose.databases.yml down
        ;;
    restart)
        $0 stop
        sleep 5
        $0 start
        ;;
    status)
        echo "Service Status:"
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        ;;
    logs)
        service=$2
        if [ -z "$service" ]; then
            echo "Usage: $0 logs <service_name>"
        else
            docker logs -f csp_$service
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs <service>}"
        exit 1
        ;;
esac
