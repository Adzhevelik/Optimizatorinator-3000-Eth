import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  ComposedChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

const PRESETS = {
  light: {
    label: 'Легкий',
    gas_estimate: 50_000,
    code: `// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IERC20 {
    function transfer(address to, uint256 amount)
        external returns (bool);
    function balanceOf(address account)
        external view returns (uint256);
}

contract SimpleTransfer {
    address public owner;

    constructor() { owner = msg.sender; }

    function transfer(
        address token,
        address to,
        uint256 amount
    ) external returns (bool) {
        require(msg.sender == owner, "Not owner");
        return IERC20(token).transfer(to, amount);
    }

    function getBalance(address token)
        external view returns (uint256) {
        return IERC20(token).balanceOf(address(this));
    }
}`,
  },
  medium: {
    label: 'Средний',
    gas_estimate: 200_000,
    code: `// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract NFTCollection is ERC721, Ownable {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIds;
    uint256 public constant MAX_SUPPLY = 10000;
    uint256 public mintPrice = 0.05 ether;
    string  private _baseTokenURI;
    mapping(address => uint256) public mintedPerWallet;
    uint256 public constant MAX_PER_WALLET = 5;

    constructor(string memory baseURI)
        ERC721("NFTCollection", "NFTC") {
        _baseTokenURI = baseURI;
    }

    function mint(uint256 qty) external payable {
        require(_tokenIds.current() + qty <= MAX_SUPPLY);
        require(msg.value >= mintPrice * qty);
        require(mintedPerWallet[msg.sender] + qty <= MAX_PER_WALLET);
        for (uint256 i = 0; i < qty; i++) {
            _tokenIds.increment();
            _safeMint(msg.sender, _tokenIds.current());
            mintedPerWallet[msg.sender]++;
        }
    }

    function withdraw() external onlyOwner {
        payable(owner()).transfer(address(this).balance);
    }
}`,
  },
  heavy: {
    label: 'Тяжелый',
    gas_estimate: 800_000,
    code: `// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/math/Math.sol";

contract DEXPool is ReentrancyGuard {
    IERC20  public tokenA;
    IERC20  public tokenB;
    uint256 public reserveA;
    uint256 public reserveB;
    uint256 public totalLiquidity;
    mapping(address => uint256) public liquidity;
    uint256 private constant FEE_NUM = 997;
    uint256 private constant FEE_DEN = 1000;

    constructor(address _a, address _b) {
        tokenA = IERC20(_a);
        tokenB = IERC20(_b);
    }

    function addLiquidity(uint256 amtA, uint256 amtB)
        external nonReentrant returns (uint256 lp) {
        tokenA.transferFrom(msg.sender, address(this), amtA);
        tokenB.transferFrom(msg.sender, address(this), amtB);
        lp = totalLiquidity == 0
            ? Math.sqrt(amtA * amtB)
            : Math.min(
                amtA * totalLiquidity / reserveA,
                amtB * totalLiquidity / reserveB);
        liquidity[msg.sender] += lp;
        totalLiquidity += lp;
        reserveA += amtA;
        reserveB += amtB;
    }

    function swap(address tokenIn, uint256 amtIn)
        external nonReentrant returns (uint256 amtOut) {
        bool isA = tokenIn == address(tokenA);
        (uint256 rIn, uint256 rOut) = isA
            ? (reserveA, reserveB) : (reserveB, reserveA);
        uint256 fee = amtIn * FEE_NUM;
        amtOut = fee * rOut / (rIn * FEE_DEN + fee);
        IERC20(tokenIn).transferFrom(msg.sender, address(this), amtIn);
        if (isA) {
            tokenB.transfer(msg.sender, amtOut);
            reserveA += amtIn; reserveB -= amtOut;
        } else {
            tokenA.transfer(msg.sender, amtOut);
            reserveB += amtIn; reserveA -= amtOut;
        }
    }
}`,
  },
};

const ACTION_GAS = {
  Swap:       150_000,
  'NFT Sale':  80_000,
  Bridging:   200_000,
  Borrowing:  300_000,
};

const TIMEZONES = [
  { value: 'UTC',                 label: 'UTC' },
  { value: 'Europe/Moscow',       label: 'Москва (UTC+3)' },
  { value: 'Europe/London',       label: 'Лондон (UTC+0/+1)' },
  { value: 'America/New_York',    label: 'Нью-Йорк (UTC-5/-4)' },
  { value: 'America/Los_Angeles', label: 'Лос-Анджелес (UTC-8/-7)' },
  { value: 'Asia/Tokyo',          label: 'Токио (UTC+9)' },
  { value: 'Asia/Singapore',      label: 'Сингапур (UTC+8)' },
  { value: 'Asia/Dubai',          label: 'Дубай (UTC+4)' },
];

function viridis(t) {
  const stops = [
    [68,  1,   84],
    [59,  82,  139],
    [33,  145, 140],
    [94,  201, 98],
    [253, 231, 37],
  ];
  const i  = Math.min(Math.floor(t * 4), 3);
  const f  = t * 4 - i;
  const c0 = stops[i];
  const c1 = stops[i + 1];
  return `rgb(${Math.round(c0[0]+f*(c1[0]-c0[0]))},${Math.round(c0[1]+f*(c1[1]-c0[1]))},${Math.round(c0[2]+f*(c1[2]-c0[2]))})`;
}

function Heatmap({ matrix, days, hours }) {
  if (!matrix || !days.length) return <div className="chart-empty">Загрузка...</div>;
  const flat = matrix.flat().filter(v => v > 0);
  const min  = Math.min(...flat);
  const max  = Math.max(...flat);
  return (
    <div className="heatmap-wrap">
      <div className="hm-axes">
        <div className="hm-ylabels">
          {days.map(d => <div key={d} className="hm-ylabel">{d}</div>)}
        </div>
        <div className="hm-grid-col">
          <div className="hm-xlabels">
            {hours.filter((_, i) => i % 3 === 0).map(h => (
              <div key={h} className="hm-xlabel">{h}</div>
            ))}
          </div>
          <div className="hm-grid">
            {days.map((day, di) => (
              <div key={day} className="hm-row">
                {hours.map((_, hi) => {
                  const v = matrix[di][hi];
                  const t = max > min ? (v - min) / (max - min) : 0;
                  return (
                    <div key={hi} className="hm-cell"
                      style={{ background: v > 0 ? viridis(t) : '#f0f0f0' }}
                      title={`${day} ${hi}:00 — ${v.toFixed(1)} Gwei`}
                    />
                  );
                })}
              </div>
            ))}
          </div>
          <div className="hm-legend-row">
            <span className="hm-lg-val">{min.toFixed(0)}</span>
            <div className="hm-lg-bar" />
            <span className="hm-lg-val">{max.toFixed(0)} Gwei</span>
          </div>
        </div>
      </div>
    </div>
  );
}

const chartTooltipStyle = {
  background: '#fff',
  border: '1px solid #e5e7eb',
  borderRadius: 2,
  fontFamily: 'monospace',
  fontSize: 11,
};

export default function App() {
  const [code,         setCode]         = useState(PRESETS.medium.code);
  const [gasEstimate,  setGasEstimate]  = useState(200_000);
  const [txPerDay,     setTxPerDay]     = useState(100);
  const [timezone,     setTimezone]     = useState('UTC');
  const [activePreset, setActivePreset] = useState('medium');
  const [result,       setResult]       = useState(null);
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState(null);
  const [netStats,     setNetStats]     = useState(null);
  const [heatmap,      setHeatmap]      = useState(null);
  const [signal,       setSignal]       = useState(null);
  const [bestMinute,   setBestMinute]   = useState(null);

  const selectPreset = key => {
    setActivePreset(key);
    setGasEstimate(PRESETS[key].gas_estimate);
    setCode(PRESETS[key].code);
  };

  const fetchNet = useCallback(async () => {
    try {
      const { data } = await axios.get(`${API_URL}/api/network-stats`);
      setNetStats(data);
    } catch {}
  }, []);

  useEffect(() => {
    fetchNet();
    axios.get(`${API_URL}/api/heatmap`).then(r => setHeatmap(r.data)).catch(() => {});
    axios.get(`${API_URL}/api/gas-signal`).then(r => setSignal(r.data)).catch(() => {});
    const iv = setInterval(fetchNet, 30_000);
    return () => clearInterval(iv);
  }, [fetchNet]);

  const predict = async () => {
    setLoading(true);
    setError(null);
    setBestMinute(null);
    try {
      const { data } = await axios.post(
        `${API_URL}/api/optimal-deployment`,
        null,
        { params: { gas_estimate: gasEstimate, tx_per_day: txPerDay } }
      );
      setResult(data);
      if (data.best_option?.deploy_in_hours) {
        try {
          const bm = await axios.get(`${API_URL}/api/best-minute`, {
            params: {
              best_hour: data.best_option.deploy_in_hours,
              timezone,
            },
          });
          setBestMinute(bm.data);
        } catch {}
      }
    } catch (e) {
      setError(e.response?.data?.detail || 'Ошибка подключения к API');
    } finally {
      setLoading(false);
    }
  };

  const allOptions = result?.all_options || [];
  const best       = result?.best_option || null;
  const tokenPrice = result?.token_price_usd || 2000;
  const currentGas = netStats?.propose_gas ?? result?.current_gas_price_gwei ?? 0;

  const chartData = [...allOptions]
    .sort((a, b) => a.deploy_in_hours - b.deploy_in_hours)
    .map(o => ({
      hour:  o.deploy_in_hours,
      point: +o.predicted_gas_price_gwei.toFixed(2),
      lower: +(o.predicted_lower_gwei ?? o.predicted_gas_price_gwei * 0.9).toFixed(2),
      upper: +(o.predicted_upper_gwei ?? o.predicted_gas_price_gwei * 1.1).toFixed(2),
    }));

  const actions = Object.entries(ACTION_GAS).map(([name, gas]) => ({
    name,
    low:  gas * (netStats?.safe_gas  ?? currentGas * 0.8) * 1e-9 * tokenPrice,
    avg:  gas *  currentGas                               * 1e-9 * tokenPrice,
    high: gas * (netStats?.fast_gas  ?? currentGas * 1.2) * 1e-9 * tokenPrice,
  }));

  const alertMsg =
    currentGas > 30 ? 'Open taps: probably too many'
  : currentGas > 15 ? 'Network load: moderate'
  : currentGas >  0 ? 'Network calm — optimal deployment window'
  : 'Ожидание данных сети...';

  const alertDark = currentGas > 15 || currentGas === 0;
  const now        = new Date();
  const refreshStr = `Last Refreshed: ${now.toUTCString().replace('GMT','UTC')}`;
  const lineCount  = code.split('\n').length;

  const bestTimeLabel = bestMinute
    ? `${bestMinute.target_hour}:00 – ${bestMinute.target_hour}:59 · ${bestMinute.timezone ?? timezone}`
    : '';

  return (
    <div className="app">

      <header className="hdr">
        <span className="hdr-title">ETH-OPTIMIZATOR-300</span>
      </header>

      <div className="top-wrap">
        <div className="editor-panel">
          <div className="editor-topbar">
            <span className="ed-fname">
              {activePreset ? `${activePreset}_contract.sol` : 'custom_contract.sol'}
            </span>
            <span className="ed-lang">Solidity</span>
          </div>
          <div className="editor-body">
            <div className="editor-gutter">
              {Array.from({ length: lineCount }, (_, i) => (
                <div key={i} className="ed-ln">{i + 1}</div>
              ))}
            </div>
            <textarea
              className="editor-code"
              value={code}
              onChange={e => { setCode(e.target.value); setActivePreset(null); }}
              spellCheck={false}
              autoComplete="off"
            />
          </div>
          <div className="editor-footer">
            <span className="ed-plus">+</span>
          </div>
        </div>

        <div className="controls-panel">
          <div className="preset-stack">
            {Object.entries(PRESETS).map(([key, p]) => (
              <button
                key={key}
                className={`pvbtn ${activePreset === key ? 'pvbtn-active' : ''}`}
                onClick={() => selectPreset(key)}
              >
                {p.label} контракт
              </button>
            ))}
          </div>
          <div className="controls-body">
            <div className="cf">
              <label className="cf-label">gas estimate</label>
              <input
                type="number"
                className="cf-input"
                value={gasEstimate}
                onChange={e => { setGasEstimate(+e.target.value || 0); setActivePreset(null); }}
              />
            </div>
            <div className="cf">
              <label className="cf-label">транзакций в день</label>
              <input
                type="number"
                className="cf-input"
                value={txPerDay}
                onChange={e => setTxPerDay(+e.target.value || 0)}
              />
            </div>
            <div className="cf">
              <label className="cf-label">часовой пояс</label>
              <select
                className="cf-input"
                value={timezone}
                onChange={e => setTimezone(e.target.value)}
              >
                {TIMEZONES.map(tz => (
                  <option key={tz.value} value={tz.value}>{tz.label}</option>
                ))}
              </select>
            </div>
          </div>
          <button
            className={`run-btn ${loading ? 'run-loading' : ''}`}
            onClick={predict}
            disabled={loading || !gasEstimate}
          >
            {loading ? 'Анализ...' : 'Получить прогноз'}
          </button>
          {error && <div className="err-bar">{error}</div>}
        </div>
      </div>

      <div className={`alert-bar ${alertDark ? 'ab-dark' : 'ab-ok'}`}>
        {alertMsg}
      </div>

      <div className="main-wrap">

        <div className="main-row-chart">
          <div className="lc-block">
            <div className="lc-head">
              <span className="lc-title">Line chart — почасовой прогноз</span>
              <span className="lc-sub">медиана · P10 · P90 · 24ч вперёд</span>
            </div>
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 24, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
                  <XAxis
                    dataKey="hour"
                    tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#9ca3af' }}
                    tickFormatter={v => `${v}h`}
                    interval={0}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#9ca3af' }}
                    domain={['auto', 'auto']}
                    label={{ value: 'Gwei', angle: -90, position: 'insideLeft', offset: 14,
                      style: { fontSize: 10, fontFamily: 'monospace', fill: '#9ca3af' } }}
                  />
                  <Tooltip
                    contentStyle={chartTooltipStyle}
                    formatter={(v, name) => {
                      const m = { point: 'Медиана', lower: 'P10', upper: 'P90' };
                      return [`${v} Gwei`, m[name] || name];
                    }}
                    labelFormatter={h => `Через ${h}ч`}
                  />
                  {best && (
                    <ReferenceLine x={best.deploy_in_hours} stroke="#0ea5a0"
                      strokeWidth={1.5} strokeDasharray="4 3"
                      label={{ value: 'BEST', position: 'top', fontSize: 9,
                        fontFamily: 'monospace', fill: '#0ea5a0' }} />
                  )}
                  <Line type="monotone" dataKey="upper"
                    stroke="rgba(14,165,160,0.3)" strokeWidth={1}
                    strokeDasharray="4 3" dot={false} name="upper" />
                  <Line type="monotone" dataKey="point"
                    stroke="#0ea5a0" strokeWidth={2}
                    dot={{ r: 2.5, fill: '#0ea5a0', strokeWidth: 0 }}
                    activeDot={{ r: 5, fill: '#0ea5a0', stroke: '#fff', strokeWidth: 2 }}
                    isAnimationActive={false}
                    name="point" />
                  <Line type="monotone" dataKey="lower"
                    stroke="rgba(230,100,50,0.4)" strokeWidth={1}
                    strokeDasharray="4 3" dot={false} name="lower" />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <div className="chart-empty">Запустите анализ для отображения данных</div>
            )}
          </div>

          <div className="forecast-block">
            <div className="fb-head">
              {['ЧЕРЕЗ (Ч)', 'ГАЗ (GWEI)', 'ДЕПЛОЙ (USD)', 'СУТКИ (USD)', 'ЭКОНОМИЯ', 'РЕК.'].map(h => (
                <div key={h} className="fb-th">{h}</div>
              ))}
            </div>
            {[...allOptions]
              .sort((a, b) => a.deploy_in_hours - b.deploy_in_hours)
              .map((o, i) => {
                const isOpt  = best?.deploy_in_hours === o.deploy_in_hours;
                const isGood = result?.current_gas_price_gwei
                  ? o.predicted_gas_price_gwei < result.current_gas_price_gwei : false;
                return (
                  <div key={i} className={`fb-row ${isOpt ? 'fb-best' : ''}`}>
                    <div>{o.deploy_in_hours}ч</div>
                    <div className="fb-v">{o.predicted_gas_price_gwei}</div>
                    <div className="fb-v">${o.deployment_cost_usd?.toFixed(4)}</div>
                    <div className="fb-v">${o.daily_operational_cost_usd?.toFixed(2)}</div>
                    <div className={o.savings_vs_current_pct > 0 ? 'fb-pos' : 'fb-neg'}>
                      {o.savings_vs_current_pct > 0 ? '+' : ''}
                      {o.savings_vs_current_pct?.toFixed(1)}%
                    </div>
                    <div>
                      {isOpt  && <span className="bdg bdg-best">BEST</span>}
                      {!isOpt && isGood  && <span className="bdg bdg-good">GOOD</span>}
                      {!isOpt && !isGood && <span className="bdg bdg-wait">wait</span>}
                    </div>
                  </div>
                );
              })}
          </div>
        </div>

        {bestMinute && bestMinute.minute_data?.length > 0 && (
          <div className="main-row-minute">
            <div className="lc-block">
              <div className="lc-head">
                <span className="lc-title">
                  Минутный прогноз · {bestTimeLabel}
                </span>
                <span className="lc-sub">
                  Сейчас {bestMinute.current_time ?? ''} · внутри оптимального часа
                </span>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <ComposedChart
                  data={bestMinute.minute_data}
                  margin={{ top: 10, right: 20, bottom: 24, left: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
                  <XAxis
                    dataKey="minute"
                    tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#9ca3af' }}
                    tickFormatter={v => `${bestMinute.target_hour}:${String(v).padStart(2,'0')}`}
                    interval={4}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fontFamily: 'monospace', fill: '#9ca3af' }}
                    domain={['auto', 'auto']}
                    label={{ value: 'Gwei', angle: -90, position: 'insideLeft', offset: 14,
                      style: { fontSize: 10, fontFamily: 'monospace', fill: '#9ca3af' } }}
                  />
                  <Tooltip
                    contentStyle={chartTooltipStyle}
                    formatter={v => [`${v} Gwei`, 'Средний газ']}
                    labelFormatter={m =>
                      `${bestMinute.target_hour}:${String(m).padStart(2,'0')}`}
                  />
                  <ReferenceLine
                    x={bestMinute.best_minute}
                    stroke="#0ea5a0"
                    strokeWidth={1.5}
                    strokeDasharray="4 3"
                    label={{ value: 'BEST', position: 'top', fontSize: 9,
                      fontFamily: 'monospace', fill: '#0ea5a0' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="avg_gwei"
                    stroke="#0ea5a0"
                    strokeWidth={2}
                    dot={(props) => {
                      const { cx, cy, payload } = props;
                      const isBest = payload.minute === bestMinute.best_minute;
                      return (
                        <circle key={payload.minute} cx={cx} cy={cy}
                          r={isBest ? 5 : 2.5}
                          fill={isBest ? '#fff' : '#0ea5a0'}
                          stroke="#0ea5a0"
                          strokeWidth={isBest ? 2 : 0}
                        />
                      );
                    }}
                    activeDot={{ r: 5, fill: '#0ea5a0', stroke: '#fff', strokeWidth: 2 }}
                    isAnimationActive={false}
                    name="avg_gwei"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            <div className="forecast-block">
              <div className="fb-head-5">
                {['ВРЕМЯ', 'МИН.', 'СРЕДНИЙ ГАЗ', 'VS ЛУЧШАЯ', 'СТАТУС'].map(h => (
                  <div key={h} className="fb-th">{h}</div>
                ))}
              </div>
              {[...bestMinute.minute_data]
                .sort((a, b) => a.minute - b.minute)
                .map((m, i) => {
                  const isBest  = m.minute === bestMinute.best_minute;
                  const diffPct = bestMinute.avg_gwei > 0
                    ? (m.avg_gwei - bestMinute.avg_gwei) / bestMinute.avg_gwei * 100
                    : 0;
                  return (
                    <div key={i} className={`fb-row-5 ${isBest ? 'fb-best' : ''}`}>
                      <div>{bestMinute.target_hour}:{String(m.minute).padStart(2,'0')}</div>
                      <div className="fb-v">{m.minute}</div>
                      <div className="fb-v">{m.avg_gwei} Gwei</div>
                      <div className={diffPct <= 0 ? 'fb-pos' : 'fb-neg'}>
                        {diffPct <= 0 ? '' : '+'}{diffPct.toFixed(1)}%
                      </div>
                      <div>
                        {isBest
                          ? <span className="bdg bdg-best">BEST</span>
                          : diffPct < 5
                          ? <span className="bdg bdg-good">GOOD</span>
                          : <span className="bdg bdg-wait">wait</span>}
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>
        )}

        <div className="main-row-bottom">
          <div className="hm-block">
            <div className="hm-head">
              <span className="hm-title">Heatmap</span>
              <span className="hm-sub">Описание хитмапа</span>
            </div>
            <Heatmap
              matrix={heatmap?.matrix}
              days={heatmap?.days   || []}
              hours={heatmap?.hours || []}
            />
          </div>

          <div className="info-panel">
            <div className="ip-section">
              <div className="ip-head">Additional Info</div>
              <div className="ip-stats">
                {[
                  ['LAST BLOCK',       netStats?.last_block?.toLocaleString() ?? '—'],
                  ['PENDING QUEUE',    netStats?.pending_queue?.toLocaleString() ?? '—'],
                  ['AVG BLOCK SIZE',   netStats?.avg_block_size ?? '—'],
                  ['AVG. UTILIZATION', netStats ? `${netStats.avg_utilization}%` : '—'],
                ].map(([l, v]) => (
                  <div key={l} className="ip-stat">
                    <div className="ip-stat-label">{l}</div>
                    <div className="ip-stat-val">{v}</div>
                  </div>
                ))}
              </div>
              <div className="ip-refresh">{refreshStr}</div>
            </div>

            <div className="ip-section">
              <div className="ip-head">Featured Actions</div>
              <table className="fa-tbl">
                <thead>
                  <tr>
                    <th>Action</th>
                    <th>Low</th>
                    <th>Average</th>
                    <th>High</th>
                  </tr>
                </thead>
                <tbody>
                  {actions.map(a => (
                    <tr key={a.name}>
                      <td><span className="fa-dot">⊙</span>{a.name}</td>
                      <td>${a.low.toFixed(2)}</td>
                      <td>${a.avg.toFixed(2)}</td>
                      <td>${a.high.toFixed(2)}</td>
                    </tr>
                  ))}
                  <tr className="fa-custom">
                    <td><span className="fa-dot">⊙</span>Custom Gas Limit</td>
                    <td></td><td></td><td></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

      </div>

    </div>
  );
}
