# Notification Templates for Prometheus Alerts

This document provides notification templates for Slack and PagerDuty integration with Prometheus Alertmanager.

## Slack Notification Template

### Configuration in alertmanager.yml

```yaml
slack_configs:
  - channel: '#ml-platform-alerts'
    title: |
      {{ if eq .GroupLabels.severity "critical" }}🚨 CRITICAL{{ else }}⚠️ WARNING{{ end }}: {{ .GroupLabels.alertname }}
    text: |
      *Summary:* {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}
      *Description:* {{ range .Alerts }}{{ .Annotations.description }}{{ end }}
      *Severity:* {{ .GroupLabels.severity }}
      *Component:* {{ .GroupLabels.component }}
      {{ range .Alerts }}
      {{ if .Annotations.runbook_url }}
      *Runbook:* {{ .Annotations.runbook_url }}
      {{ end }}
      {{ end }}
    color: '{{ if eq .GroupLabels.severity "critical" }}danger{{ else }}warning{{ end }}'
```

### Slack Webhook Setup

1. Create a Slack webhook:
   - Go to https://api.slack.com/apps
   - Create a new app or use existing
   - Enable Incoming Webhooks
   - Add webhook to workspace
   - Copy webhook URL

2. Set environment variable:
   ```bash
   export SLACK_API_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
   ```

## PagerDuty Notification Template

### Configuration in alertmanager.yml

```yaml
pagerduty_configs:
  - service_key: '${PAGERDUTY_SERVICE_KEY}'
    description: '{{ .GroupLabels.alertname }}'
    severity: '{{ .GroupLabels.severity }}'
    details:
      summary: '{{ .GroupLabels.alertname }}'
      description: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
      component: '{{ .GroupLabels.component }}'
      runbook_url: '{{ range .Alerts }}{{ .Annotations.runbook_url }}{{ end }}'
```

### PagerDuty Integration Setup

1. Create PagerDuty service:
   - Go to PagerDuty → Services
   - Create new service or select existing
   - Add "Prometheus" integration
   - Copy Integration Key

2. Set environment variable:
   ```bash
   export PAGERDUTY_SERVICE_KEY="your-integration-key"
   ```

## Custom Template File (Optional)

For more complex templates, you can use a custom template file:

**templates/slack.tmpl:**
```
{{ define "slack.custom.title" }}
{{ if eq .Status "firing" }}
🚨 {{ .GroupLabels.alertname }} - {{ .GroupLabels.severity }}
{{ else }}
✅ Resolved: {{ .GroupLabels.alertname }}
{{ end }}
{{ end }}

{{ define "slack.custom.text" }}
{{ range .Alerts }}
*Alert:* {{ .Annotations.summary }}
*Description:* {{ .Annotations.description }}
*Labels:* {{ range .Labels.SortedPairs }}{{ .Name }}={{ .Value }} {{ end }}
{{ end }}
{{ end }}
```

Then reference in alertmanager.yml:
```yaml
templates:
  - '/etc/alertmanager/templates/*.tmpl'
```

## Example Notification Formats

### Critical Alert (Slack)
```
🚨 CRITICAL: HighErrorRate

*Summary:* High error rate detected
*Description:* Error rate is 7.5% for the last 5 minutes. This exceeds the 5% threshold.
*Severity:* critical
*Component:* api
*Runbook:* https://wiki.example.com/runbooks/high-error-rate
*Dashboard:* http://grafana.example.com/d/recommender-overview
```

### Warning Alert (Slack)
```
⚠️ WARNING: HighLatency

*Summary:* High P95 latency detected
*Description:* P95 latency is 250ms for the last 5 minutes. This exceeds the 200ms threshold.
*Severity:* warning
*Component:* api
*Endpoint:* /recommend/user_123
*Dashboard:* http://grafana.example.com/d/recommender-overview
```

### PagerDuty Alert
- **Title:** HighErrorRate
- **Severity:** critical
- **Description:** Error rate is 7.5% for the last 5 minutes. This exceeds the 5% threshold.
- **Component:** api
- **Runbook URL:** https://wiki.example.com/runbooks/high-error-rate
- **Dashboard URL:** http://grafana.example.com/d/recommender-overview
