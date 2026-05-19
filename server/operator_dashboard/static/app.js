(() => {
  const state = {
    token: new URLSearchParams(window.location.search).get("token") || "",
    selectedServiceId: null,
    status: null,
    error: null,
    robotCommandStatus: null,
    refreshing: false,
    busy: false,
  };
  const auxiliaryServiceIds = new Set(["wake_tuning", "voice_modulation"]);

  const nodes = {
    globalStatus: document.getElementById("global-status"),
    serviceGrid: document.getElementById("service-grid"),
    serviceCount: document.getElementById("service-count"),
    auxiliaryServiceGrid: document.getElementById("auxiliary-service-grid"),
    auxiliaryServiceCount: document.getElementById("auxiliary-service-count"),
    healthList: document.getElementById("health-list"),
    logPanel: document.getElementById("log-panel"),
    startAll: document.getElementById("start-all"),
    stopAll: document.getElementById("stop-all"),
    refresh: document.getElementById("refresh"),
    copyLogs: document.getElementById("copy-logs"),
    robotHome: document.getElementById("robot-home"),
    robotSyncState: document.getElementById("robot-sync-state"),
    gripperOpen: document.getElementById("gripper-open"),
    gripperClose: document.getElementById("gripper-close"),
    robotCommandStatus: document.getElementById("robot-command-status"),
  };

  async function api(path, options = {}) {
    const url = new URL(path, window.location.origin);
    url.searchParams.set("token", state.token);

    const response = await fetch(url, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.headers || {}),
      },
    });

    if (!response.ok) {
      const text = await response.text();
      let detail = text || response.statusText;
      try {
        const body = JSON.parse(text);
        detail = body.detail || JSON.stringify(body);
      } catch (_error) {
        // Keep the plain-text detail.
      }
      throw new Error(`${response.status} ${detail}`.trim());
    }

    return response.json();
  }

  function element(tag, attributes = {}, children = []) {
    const node = document.createElement(tag);
    Object.entries(attributes).forEach(([key, value]) => {
      if (value === null || value === undefined || value === false) {
        return;
      }
      if (key === "className") {
        node.className = value;
      } else if (key === "dataset") {
        Object.entries(value).forEach(([dataKey, dataValue]) => {
          node.dataset[dataKey] = dataValue;
        });
      } else if (key === "text") {
        node.textContent = value;
      } else {
        node.setAttribute(key, value);
      }
    });

    children.forEach((child) => {
      node.append(child instanceof Node ? child : document.createTextNode(child));
    });
    return node;
  }

  function serviceEntries() {
    if (!state.status || !state.status.services) {
      return [];
    }
    return Object.entries(state.status.services).sort(([left], [right]) =>
      left.localeCompare(right),
    );
  }

  function selectedService() {
    if (!state.selectedServiceId || !state.status || !state.status.services) {
      return null;
    }
    return state.status.services[state.selectedServiceId] || null;
  }

  function selectorEscape(value) {
    const text = String(value);
    if (window.CSS && typeof window.CSS.escape === "function") {
      return window.CSS.escape(text);
    }
    return text.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
  }

  function dataAttributeSelector(name, value) {
    return `[data-${name}="${selectorEscape(value)}"]`;
  }

  function serviceGridNodes() {
    return [nodes.serviceGrid, nodes.auxiliaryServiceGrid].filter(Boolean);
  }

  function serviceGridFor(node) {
    return serviceGridNodes().find((grid) => grid.contains(node)) || null;
  }

  function captureServiceGridFocus() {
    const activeElement = document.activeElement;
    const grid = activeElement ? serviceGridFor(activeElement) : null;
    if (!grid) {
      return null;
    }

    const actionButton = activeElement.closest("[data-action][data-service-id]");
    if (actionButton && grid.contains(actionButton)) {
      return {
        serviceId: actionButton.dataset.serviceId,
        action: actionButton.dataset.action,
      };
    }

    const card = activeElement.closest(".service-card[data-service-id]");
    if (card && activeElement === card) {
      return { serviceId: card.dataset.serviceId, action: null };
    }

    return null;
  }

  function restoreServiceGridFocus(focusState) {
    if (!focusState) {
      return;
    }

    const serviceSelector = dataAttributeSelector("service-id", focusState.serviceId);
    const selector = focusState.action
      ? `${serviceSelector}${dataAttributeSelector("action", focusState.action)}`
      : `.service-card${serviceSelector}`;
    const focusTarget = serviceGridNodes()
      .map((grid) => grid.querySelector(selector))
      .find(Boolean);
    if (focusTarget) {
      focusTarget.focus();
    }
  }

  function setBusy(isBusy) {
    state.busy = isBusy;
    [
      nodes.startAll,
      nodes.stopAll,
      nodes.refresh,
      nodes.robotHome,
      nodes.robotSyncState,
      nodes.gripperOpen,
      nodes.gripperClose,
    ].forEach((button) => {
      button.disabled = isBusy;
    });
    renderRobotControls();
  }

  function stateLabel(stateName) {
    return String(stateName || "unknown").toUpperCase();
  }

  function isGlobalService(serviceId, service) {
    return (
      service.include_in_global_actions !== false &&
      !auxiliaryServiceIds.has(serviceId)
    );
  }

  function renderGlobalStatus() {
    const entries = serviceEntries().filter(([serviceId, service]) =>
      isGlobalService(serviceId, service),
    );
    nodes.globalStatus.className = "global-status";

    if (state.error) {
      nodes.globalStatus.textContent = `FAULT: ${state.error}`;
      nodes.globalStatus.classList.add("status-failed");
      return;
    }

    if (!entries.length) {
      nodes.globalStatus.textContent = "No system services configured";
      nodes.globalStatus.classList.add("status-stopped");
      return;
    }

    const counts = entries.reduce((accumulator, [, service]) => {
      accumulator[service.state] = (accumulator[service.state] || 0) + 1;
      return accumulator;
    }, {});
    const total = entries.length;
    let label = "MIXED";
    let className = "status-mixed";

    if (counts.failed) {
      label = "FAULT";
      className = "status-failed";
    } else if (counts.degraded) {
      label = "DEGRADED";
      className = "status-degraded";
    } else if (counts.starting || counts.stopping) {
      label = "TRANSITION";
      className = "status-transition";
    } else if (counts.ready === total) {
      label = "READY";
      className = "status-ready";
    } else if (counts.stopped === total) {
      label = "STOPPED";
      className = "status-stopped";
    }

    nodes.globalStatus.textContent = `${label} · ${counts.ready || 0}/${total} ready`;
    nodes.globalStatus.classList.add(className);
  }

  function safeHref(url) {
    try {
      const parsed = new URL(url, window.location.origin);
      if (["http:", "https:"].includes(parsed.protocol)) {
        return parsed.href;
      }
    } catch (_error) {
      return null;
    }
    return null;
  }

  function linkNode(label, url) {
    const href = safeHref(url);
    if (!href) {
      return null;
    }
    return element("a", {
      className: "service-link",
      href,
      target: "_blank",
      rel: "noreferrer noopener",
      text: label,
    });
  }

  function renderServiceCard(serviceId, service) {
    const isSelected = serviceId === state.selectedServiceId;
    const serviceName = service.label || serviceId;
    const badge = element("span", {
      className: `state-badge state-${service.state || "unknown"}`,
      text: stateLabel(service.state),
    });
    const auxiliaryBadge = isGlobalService(serviceId, service)
      ? null
      : element("span", {
          className: "state-badge state-auxiliary",
          text: "AUX",
        });
    const title = element("div", { className: "service-title" }, [
      element("h3", { text: serviceName }),
      element("code", { text: serviceId }),
    ]);

    const meta = element("dl", { className: "service-meta" }, [
      element("div", {}, [
        element("dt", { text: "PID" }),
        element("dd", { text: service.pid || "—" }),
      ]),
      element("div", {}, [
        element("dt", { text: "Exit" }),
        element("dd", { text: service.last_exit_code ?? "—" }),
      ]),
    ]);

    const command = element("p", {
      className: "service-command",
      text: (service.command || []).join(" ") || "No command configured",
    });

    const links = element("div", { className: "service-links" });
    (service.links || []).forEach((link) => {
      const node = linkNode(link.label || link.url, link.url);
      if (node) {
        links.append(node);
      }
    });
    (service.detected_urls || []).forEach((url) => {
      const node = linkNode("detected", url);
      if (node) {
        links.append(node);
      }
    });

    if (!links.childElementCount) {
      links.append(element("span", { className: "muted", text: "No links detected" }));
    }

    const actions = element("div", { className: "service-actions" }, [
      element("button", {
        className: "control control-small control-start",
        type: "button",
        "aria-label": `Start ${serviceName}`,
        text: "Start",
        dataset: { action: "start", serviceId },
      }),
      element("button", {
        className: "control control-small",
        type: "button",
        "aria-label": `Restart ${serviceName}`,
        text: "Restart",
        dataset: { action: "restart", serviceId },
      }),
      element("button", {
        className: "control control-small control-stop",
        type: "button",
        "aria-label": `Stop ${serviceName}`,
        text: "Stop",
        dataset: { action: "stop", serviceId },
      }),
    ]);

    return element(
      "article",
      {
        className: `service-card${isSelected ? " is-selected" : ""}${
          isGlobalService(serviceId, service) ? "" : " is-auxiliary"
        }`,
        tabindex: "0",
        dataset: { serviceId },
      },
      [
        element("div", { className: "service-card-top" }, [
          title,
          element("div", { className: "badge-row" }, [
            ...(auxiliaryBadge ? [auxiliaryBadge] : []),
            badge,
          ]),
        ]),
        meta,
        command,
        links,
        service.last_error
          ? element("p", { className: "service-error", text: service.last_error })
          : element("span"),
        actions,
      ],
    );
  }

  function renderServices() {
    const entries = serviceEntries();
    const coreEntries = entries.filter(([serviceId, service]) =>
      isGlobalService(serviceId, service),
    );
    const auxiliaryEntries = entries.filter(
      ([serviceId, service]) => !isGlobalService(serviceId, service),
    );
    if (
      entries.length &&
      (!state.selectedServiceId || !state.status.services[state.selectedServiceId])
    ) {
      state.selectedServiceId = (coreEntries[0] || auxiliaryEntries[0])[0];
    }

    const focusState = captureServiceGridFocus();
    nodes.serviceGrid.replaceChildren(
      ...coreEntries.map(([serviceId, service]) =>
        renderServiceCard(serviceId, service),
      ),
    );
    nodes.auxiliaryServiceGrid.replaceChildren(
      ...auxiliaryEntries.map(([serviceId, service]) =>
        renderServiceCard(serviceId, service),
      ),
    );
    restoreServiceGridFocus(focusState);

    const readyCount = coreEntries.filter(
      ([, service]) => service.state === "ready",
    ).length;
    nodes.serviceCount.textContent = `${coreEntries.length} core · ${readyCount} ready`;
    nodes.auxiliaryServiceCount.textContent = `${auxiliaryEntries.length} auxiliary`;
  }

  function renderHealth() {
    const service = selectedService();
    const checks = service ? service.ready_checks || [] : [];

    if (!service) {
      nodes.healthList.replaceChildren(
        element("li", { className: "muted", text: "No service selected" }),
      );
      return;
    }

    if (!checks.length) {
      nodes.healthList.replaceChildren(
        element("li", { className: "muted", text: "No readiness checks configured" }),
      );
      return;
    }

    nodes.healthList.replaceChildren(
      ...checks.map((check) =>
        element("li", { className: `health-item ${healthCheckClassName(check)}` }, [
          element("span", {
            className: "health-label",
            text: check.label || check.type || "check",
          }),
          element("span", { className: "health-detail", text: check.detail || "—" }),
        ]),
      ),
    );
  }

  function verifiedExecutionService() {
    if (!state.status || !state.status.services) {
      return null;
    }
    return state.status.services.verified_execution || null;
  }

  function healthCheckClassName(check) {
    if (check.ok) {
      return "ok";
    }
    return check.required === false ? "warn" : "fail";
  }

  function renderRobotControls() {
    const service = verifiedExecutionService();
    const isReady = service && service.state === "ready";
    const controlsDisabled = state.busy || !isReady;
    [
      nodes.robotHome,
      nodes.robotSyncState,
      nodes.gripperOpen,
      nodes.gripperClose,
    ].forEach((button) => {
      button.disabled = controlsDisabled;
    });

    if (state.robotCommandStatus) {
      nodes.robotCommandStatus.textContent = state.robotCommandStatus;
    } else if (isReady) {
      nodes.robotCommandStatus.textContent = "Verified execution ready";
    } else {
      nodes.robotCommandStatus.textContent = "Verified execution stopped";
    }
  }

  function renderLogs() {
    const service = selectedService();
    if (!service) {
      nodes.logPanel.textContent = "Select a service to inspect logs.";
      return;
    }

    const logs = service.recent_logs || [];
    nodes.logPanel.textContent = logs.length
      ? logs.join("\n")
      : `${service.label || service.id} has no recent logs.`;
  }

  function render() {
    renderGlobalStatus();
    renderServices();
    renderRobotControls();
    renderHealth();
    renderLogs();
  }

  async function refresh() {
    if (state.refreshing) {
      return;
    }
    state.refreshing = true;
    try {
      state.status = await api("/api/status");
      state.error = null;
    } catch (error) {
      state.error = error.message || String(error);
    } finally {
      state.refreshing = false;
      render();
    }
  }

  async function runAction(path) {
    let actionError = null;
    setBusy(true);
    try {
      await api(path, { method: "POST" });
      state.error = null;
    } catch (error) {
      actionError = error.message || String(error);
    } finally {
      setBusy(false);
      await refresh();
      if (actionError) {
        state.error = actionError;
        render();
      }
    }
  }

  function selectService(serviceId) {
    state.selectedServiceId = serviceId;
    render();
  }

  function handleServiceGridClick(event) {
    const actionButton = event.target.closest("[data-action]");
    if (actionButton) {
      event.stopPropagation();
      const serviceId = actionButton.dataset.serviceId;
      const action = actionButton.dataset.action;
      runAction(`/api/services/${encodeURIComponent(serviceId)}/${action}`);
      return;
    }

    const card = event.target.closest("[data-service-id]");
    if (card) {
      selectService(card.dataset.serviceId);
    }
  }

  async function runRobotCommand(path, label, messages = {}) {
    if (path === "/api/robot/home" && !window.confirm("Move the real robot to home?")) {
      return;
    }

    let actionError = null;
    setBusy(true);
    state.robotCommandStatus = messages.pending || `${label} pending`;
    renderRobotControls();
    try {
      const result = await api(path, { method: "POST" });
      state.robotCommandStatus = result.status || `${label} complete`;
      state.error = null;
    } catch (error) {
      actionError = error.message || String(error);
      state.robotCommandStatus = messages.failed || `${label} failed`;
    } finally {
      setBusy(false);
      await refresh();
      if (actionError) {
        state.error = actionError;
        render();
      }
    }
  }

  function handleServiceGridKeydown(event) {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    const card = event.target.closest("[data-service-id]");
    if (card && event.target === card) {
      event.preventDefault();
      selectService(card.dataset.serviceId);
    }
  }

  serviceGridNodes().forEach((grid) => {
    grid.addEventListener("click", handleServiceGridClick);
    grid.addEventListener("keydown", handleServiceGridKeydown);
  });

  nodes.startAll.addEventListener("click", () => runAction("/api/start-all"));
  nodes.stopAll.addEventListener("click", () => runAction("/api/stop-all"));
  nodes.refresh.addEventListener("click", refresh);
  nodes.robotHome.addEventListener("click", () =>
    runRobotCommand("/api/robot/home", "Home"),
  );
  nodes.robotSyncState.addEventListener("click", () =>
    runRobotCommand("/api/robot/sync-state", "Align MoveIt", {
      pending: "Align MoveIt pending",
      failed: "Align MoveIt failed",
    }),
  );
  nodes.gripperOpen.addEventListener("click", () =>
    runRobotCommand("/api/robot/gripper/open", "Open gripper"),
  );
  nodes.gripperClose.addEventListener("click", () =>
    runRobotCommand("/api/robot/gripper/close", "Close gripper"),
  );
  nodes.copyLogs.addEventListener("click", async () => {
    const text = nodes.logPanel.textContent || "";
    try {
      await navigator.clipboard.writeText(text);
      nodes.copyLogs.textContent = "Copied";
      setTimeout(() => {
        nodes.copyLogs.textContent = "Copy logs";
      }, 1200);
    } catch (_error) {
      nodes.copyLogs.textContent = "Copy failed";
      setTimeout(() => {
        nodes.copyLogs.textContent = "Copy logs";
      }, 1200);
    }
  });

  refresh();
  window.setInterval(refresh, 2000);
})();
