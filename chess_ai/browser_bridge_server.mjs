import http from 'node:http';

let task = null;
let page = null;
let gameInProgress = false;

const PORT = 45678;
const COLS = ["a", "b", "c", "d", "e", "f", "g", "h", "i"];

async function getSquareCenter(col, row) {
  const colLetter = COLS[col];
  const rankNumber = row + 1;
  const selector = `.square[class*='${rankNumber}-${colLetter}']`;
  const rect = await page.evaluate((sel) => {
    const el = document.querySelector(sel);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  }, selector);
  return rect;
}

async function ensurePage() {
  if (!task) {
    task = await taskSpace("xiangqi online training");
    page = task.page("p1");
  }
  return page;
}

async function startNewGame(level = 1) {
  const p = await ensurePage();
  await p.goto("https://play.xiangqi.com/computer", { waitUntil: "domcontentloaded" });
  await p.waitForTimeout(2000);

  // Select Bot Level (1 to 10)
  const targetLevel = Math.max(1, Math.min(10, Number(level) || 1));
  const botInfo = await p.evaluate((lvl) => {
    const buttons = document.querySelectorAll(".all-bots li button.bot-item");
    if (buttons[lvl - 1]) {
      buttons[lvl - 1].click();
    }
    const detail = document.querySelector(".bot-detail");
    return detail ? detail.innerText.trim() : "";
  }, targetLevel);
  await p.waitForTimeout(500);

  // Select Red
  await p.evaluate(() => {
    const sideWrapper = document.querySelector(".side-wrapper");
    if (sideWrapper) {
      const labels = Array.from(sideWrapper.querySelectorAll("label, input, div"));
      const redLabel = labels.find(el => el.innerText && el.innerText.trim() === "Red");
      if (redLabel) redLabel.click();
    }
  });
  await p.waitForTimeout(500);

  // Click Play
  await p.evaluate(() => {
    const playBtn = document.querySelector(".play-custom-btn");
    if (playBtn) playBtn.click();
  });
  await p.waitForTimeout(2000);

  gameInProgress = true;
  return { status: "ready", bot: botInfo || `Level ${targetLevel}`, level: targetLevel };
}

async function makeMove(fromCol, fromRow, toCol, toRow, expectedMoveCount) {
  const p = await ensurePage();
  const fromRect = await getSquareCenter(fromCol, fromRow);
  const toRect = await getSquareCenter(toCol, toRow);
  if (!fromRect || !toRect) {
    throw new Error(`Squares not found: from (${fromCol}, ${fromRow}) to (${toCol}, ${toRow})`);
  }

  // Click from square
  await p.mouse.click(fromRect.x, fromRect.y);
  await p.waitForTimeout(500);

  // Click to square
  await p.mouse.click(toRect.x, toRect.y);
  await p.waitForTimeout(1000);

  // Wait for opponent's response
  const startTime = Date.now();
  let latestOpponentMove = null;
  let allMoves = [];
  let gameOver = null;

  while (Date.now() - startTime < 20000) {
    const state = await p.evaluate(() => {
      const moves = Array.from(document.querySelectorAll("button.move")).map(b => b.innerText.trim()).filter(Boolean);
      // Check for checkmate / resign / draw text
      const modal = document.querySelector("[class*='modal'], [class*='dialog'], [class*='result-modal'], [class*='game-over']");
      const modalText = modal ? modal.innerText.trim() : "";
      
      const isRedTurn = !!document.querySelector(".playerWrapper.bottom .online-dot, .playerWrapper.bottom [class*='active'], .playerWrapper.bottom [style*='green']");

      return { moves, modalText, isRedTurn };
    });

    allMoves = state.moves;
    if (state.modalText && (state.modalText.toLowerCase().includes("win") || state.modalText.toLowerCase().includes("draw") || state.modalText.toLowerCase().includes("checkmate"))) {
      gameOver = state.modalText;
      break;
    }

    if (allMoves.length >= expectedMoveCount) {
      latestOpponentMove = allMoves[allMoves.length - 1];
      break;
    }

    await p.waitForTimeout(400);
  }

  return {
    status: "ok",
    moves: allMoves,
    latestOpponentMove,
    gameOver
  };
}

async function closeAll() {
  if (task) {
    try {
      await task.finish({ keep: [] });
    } catch (e) {
      console.error("Error closing task:", e);
    }
    task = null;
    page = null;
  }
}

const server = http.createServer(async (req, res) => {
  res.setHeader("Content-Type", "application/json");

  let body = "";
  req.on("data", chunk => { body += chunk; });
  req.on("end", async () => {
    try {
      const data = body ? JSON.parse(body) : {};

      if (req.url === "/ping") {
        res.writeHead(200);
        res.end(JSON.stringify({ status: "pong" }));
      } else if (req.url === "/start_game") {
        const result = await startNewGame(data.level || 1);
        res.writeHead(200);
        res.end(JSON.stringify(result));
      } else if (req.url === "/play_move") {
        const result = await makeMove(data.from_col, data.from_row, data.to_col, data.to_row, data.expected_move_count);
        res.writeHead(200);
        res.end(JSON.stringify(result));
      } else if (req.url === "/close") {
        await closeAll();
        res.writeHead(200);
        res.end(JSON.stringify({ status: "closed" }));
        server.close();
        process.exit(0);
      } else {
        res.writeHead(404);
        res.end(JSON.stringify({ error: "not_found" }));
      }
    } catch (err) {
      console.error("Request error:", err);
      res.writeHead(500);
      res.end(JSON.stringify({ error: String(err) }));
    }
  });
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`BROWSER_BRIDGE_READY_${PORT}`);
});
