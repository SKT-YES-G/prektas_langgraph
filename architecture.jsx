import { useState } from "react";

const NODE_COLORS = {
  stream: { bg: "#0c1a2e", border: "#38bdf8", text: "#93c5fd" },
  router: { bg: "#1a0d38", border: "#a78bfa", text: "#c4b5fd" },
  s2: { bg: "#0d1f40", border: "#3b82f6", text: "#93c5fd" },
  s3: { bg: "#0d2818", border: "#22c55e", text: "#86efac" },
  s4: { bg: "#2a1000", border: "#f97316", text: "#fdba74" },
  question: { bg: "#1c1400", border: "#fbbf24", text: "#fde68a", dashed: true },
  candidate: { bg: "#13103a", border: "#818cf8", text: "#a5b4fc", dashed: true },
  output: { bg: "#1a0a20", border: "#ec4899", text: "#f9a8d4" },
  state: { bg: "#0f172a", border: "#334155", text: "#94a3b8", dashed: true },
};

function Box({ x, y, w, h, color, title, sub, sub2, emoji, rx = 10 }) {
  const c = NODE_COLORS[color];
  const dash = c.dashed ? "5,3" : "none";
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={rx}
        fill={c.bg} stroke={c.border} strokeWidth={2} strokeDasharray={dash} />
      <text x={x + w / 2} y={y + (sub ? (sub2 ? 20 : 22) : 28)} textAnchor="middle"
        fontSize={12.5} fontWeight="700" fill={c.text}>
        {emoji && <tspan>{emoji} </tspan>}{title}
      </text>
      {sub && <text x={x + w / 2} y={y + (sub2 ? 38 : 42)} textAnchor="middle" fontSize={9.5} fill="#64748b">{sub}</text>}
      {sub2 && <text x={x + w / 2} y={y + 54} textAnchor="middle" fontSize={9} fill="#4a5568">{sub2}</text>}
    </g>
  );
}

function Arrow({ d, color, dashed }) {
  const colors = {
    sky: "#38bdf8", purple: "#a78bfa", blue: "#60a5fa",
    green: "#4ade80", orange: "#fb923c", yellow: "#fbbf24",
    indigo: "#818cf8", pink: "#f472b6", gray: "#475569"
  };
  return (
    <path d={d} fill="none" stroke={colors[color] || color}
      strokeWidth={1.8} strokeDasharray={dashed ? "4,3" : "none"}
      markerEnd={`url(#ah-${color})`} />
  );
}

function Label({ x, y, text, color = "#64748b", size = 9.5, anchor = "middle", rotate }) {
  return (
    <text x={x} y={y} textAnchor={anchor} fontSize={size} fill={color}
      transform={rotate ? `rotate(${rotate[0]}, ${rotate[1]}, ${rotate[2]})` : undefined}>
      {text}
    </text>
  );
}

export default function PreKTASFlow() {
  const [hovered, setHovered] = useState(null);

  const markers = ["sky","purple","blue","green","orange","yellow","indigo","pink","gray"];
  const markerColors = {
    sky:"#38bdf8",purple:"#a78bfa",blue:"#60a5fa",green:"#4ade80",
    orange:"#fb923c",yellow:"#fbbf24",indigo:"#818cf8",pink:"#f472b6",gray:"#475569"
  };

  return (
    <div style={{ background: "#0a0e17", minHeight: "100vh", padding: "28px 16px", fontFamily: "'Segoe UI', sans-serif" }}>
      <h1 style={{ textAlign: "center", fontSize: 18, fontWeight: 800, color: "#f1f5f9", marginBottom: 4 }}>
        Pre-KTAS Classification Agent — Flow Diagram
      </h1>
      <p style={{ textAlign: "center", fontSize: 11, color: "#475569", marginBottom: 24 }}>
        n초마다 streaming input → 재평가 판단 노드 → Stage 2 / 3 / 4 분류 파이프라인 → 최종 KTAS 등급
      </p>

      <svg width="100%" viewBox="0 0 1040 760" style={{ display: "block", margin: "0 auto", maxWidth: 1040 }}>
        <defs>
          {markers.map(m => (
            <marker key={m} id={`ah-${m}`} markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
              <polygon points="0 0, 8 3, 0 6" fill={markerColors[m]} />
            </marker>
          ))}
          <filter id="glow">
            <feGaussianBlur stdDeviation="2.5" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        {/* ── STREAMING INPUT ── */}
        <g filter="url(#glow)">
          <Box x={375} y={14} w={220} h={56} color="stream" emoji="🔄" title="Streaming Input" sub="n초마다 새 정보 유입 (음성 / 키보드)" />
        </g>

        {/* ── STATE ── */}
        <rect x={22} y={105} width={140} height={70} rx={8} fill="#0f172a" stroke="#334155" strokeWidth={1.2} strokeDasharray="3,2"/>
        <text x={92} y={123} textAnchor="middle" fontSize={10} fill="#64748b">📦 State</text>
        <text x={92} y={138} textAnchor="middle" fontSize={9} fill="#94a3b8">• 이전 분류 결과</text>
        <text x={92} y={152} textAnchor="middle" fontSize={9} fill="#94a3b8">• 사용자 입력 이력</text>
        <text x={92} y={166} textAnchor="middle" fontSize={9} fill="#94a3b8">• 키보드 답변</text>
        <line x1={162} y1={144} x2={372} y2={144} stroke="#334155" strokeWidth={1.2} strokeDasharray="3,2" markerEnd="url(#ah-gray)"/>

        {/* stream → reeval */}
        <line x1={485} y1={70} x2={485} y2={110} stroke="#38bdf8" strokeWidth={2} markerEnd="url(#ah-sky)"/>
        <text x={491} y={96} fontSize={9.5} fill="#38bdf8">streaming + state</text>

        {/* ── 재평가 판단 노드 ── */}
        <g filter="url(#glow)">
          <Box x={368} y={112} w={234} h={60} rx={12} color="router" emoji="⚖️" title="재평가 판단 노드" sub="LLM 라우터: Stage 2 / 3 / 4 재평가 분기" />
        </g>

        {/* branch → Stage2 */}
        <path d="M 388 172 C 318 172 250 228 188 228" fill="none" stroke="#60a5fa" strokeWidth={2} markerEnd="url(#ah-blue)"/>
        <text x={256} y={188} fontSize={9.5} fill="#60a5fa">Stage 2 재평가</text>

        {/* branch → Stage3 */}
        <line x1={485} y1={172} x2={485} y2={228} stroke="#a78bfa" strokeWidth={2} markerEnd="url(#ah-purple)"/>
        <text x={490} y={209} fontSize={9.5} fill="#a78bfa">Stage 3 재평가</text>

        {/* branch → Stage4 */}
        <path d="M 582 172 C 660 172 754 228 806 228" fill="none" stroke="#fb923c" strokeWidth={2} markerEnd="url(#ah-orange)"/>
        <text x={666} y={188} fontSize={9.5} fill="#fb923c">Stage 4 재평가</text>

        {/* ── STAGE 재평가 노드 3개 ── */}
        <Box x={58} y={230} w={210} h={58} color="s2" title="Stage 2 재평가 노드" sub="추가 질문 생성 or S2 분류기" />
        <Box x={374} y={230} w={222} h={58} color="s3" title="Stage 3 재평가 노드" sub="추가 질문 생성 or S3 분류기" />
        <Box x={714} y={230} w={210} h={58} color="s4" title="Stage 4 재평가 노드" sub="추가 질문 생성 or S4 분류기" />

        {/* ── 추가 질문 박스 ── */}
        {/* S2 */}
        <rect x={8} y={354} width={138} height={46} rx={8} fill="#1c1400" stroke="#fbbf24" strokeWidth={1.6} strokeDasharray="5,3"/>
        <text x={77} y={374} textAnchor="middle" fontSize={11} fontWeight={700} fill="#fde68a">💬 추가 질문 생성</text>
        <text x={77} y={390} textAnchor="middle" fontSize={9.5} fill="#78716c">사용자에게 전송</text>
        <path d="M 120 288 C 94 322 82 338 82 354" fill="none" stroke="#fbbf24" strokeWidth={1.5} strokeDasharray="4,3" markerEnd="url(#ah-yellow)"/>
        <text x={40} y={328} fontSize={9.5} fill="#fbbf24">질문 생성</text>

        {/* S3 */}
        <rect x={420} y={354} width={138} height={46} rx={8} fill="#1c1400" stroke="#fbbf24" strokeWidth={1.6} strokeDasharray="5,3"/>
        <text x={489} y={374} textAnchor="middle" fontSize={11} fontWeight={700} fill="#fde68a">💬 추가 질문 생성</text>
        <text x={489} y={390} textAnchor="middle" fontSize={9.5} fill="#78716c">사용자에게 전송</text>
        <path d="M 448 288 C 438 326 436 340 444 354" fill="none" stroke="#fbbf24" strokeWidth={1.5} strokeDasharray="4,3" markerEnd="url(#ah-yellow)"/>
        <text x={400} y={326} fontSize={9.5} fill="#fbbf24">질문 생성</text>

        {/* S4 */}
        <rect x={870} y={354} width={138} height={46} rx={8} fill="#1c1400" stroke="#fbbf24" strokeWidth={1.6} strokeDasharray="5,3"/>
        <text x={939} y={374} textAnchor="middle" fontSize={11} fontWeight={700} fill="#fde68a">💬 추가 질문 생성</text>
        <text x={939} y={390} textAnchor="middle" fontSize={9.5} fill="#78716c">사용자에게 전송</text>
        <path d="M 868 288 C 896 322 920 342 920 354" fill="none" stroke="#fbbf24" strokeWidth={1.5} strokeDasharray="4,3" markerEnd="url(#ah-yellow)"/>
        <text x={876} y={326} fontSize={9.5} fill="#fbbf24">질문 생성</text>

        {/* ── CLASSIFIERS ── */}
        {/* S2 → classifier */}
        <line x1={163} y1={288} x2={163} y2={374} stroke="#3b82f6" strokeWidth={2} markerEnd="url(#ah-blue)"/>
        <text x={168} y={339} fontSize={9.5} fill="#60a5fa">S2 분류기</text>

        <Box x={68} y={376} w={190} h={68} color="s2" emoji="🗂" title="Stage 2 Classifier" sub="대분류 18개 항목 중 선택" sub2="물질오용 / 신경계 / 호흡기계 …" />

        {/* S3 → classifier */}
        <line x1={485} y1={288} x2={485} y2={374} stroke="#22c55e" strokeWidth={2} markerEnd="url(#ah-green)"/>
        <text x={490} y={339} fontSize={9.5} fill="#4ade80">S3 분류기</text>

        <Box x={380} y={376} w={210} h={68} color="s3" emoji="🗂" title="Stage 3 Classifier" sub="소분류 선택" sub2="물질오용/중독 / 과다복용 / 금단 …" />

        {/* S4 → classifier */}
        <line x1={819} y1={288} x2={819} y2={374} stroke="#f97316" strokeWidth={2} markerEnd="url(#ah-orange)"/>
        <text x={824} y={339} fontSize={9.5} fill="#fb923c">S4 분류기</text>

        <Box x={712} y={376} w={214} h={68} color="s4" emoji="🗂" title="Stage 4 Classifier" sub="증상 기반 최종 분류" sub2="→ KTAS 등급 결정" />

        {/* ── CANDIDATES MAPPING ── */}
        <line x1={163} y1={444} x2={163} y2={498} stroke="#818cf8" strokeWidth={1.8} strokeDasharray="4,2" markerEnd="url(#ah-indigo)"/>
        <rect x={72} y={500} width={182} height={50} rx={8} fill="#13103a" stroke="#818cf8" strokeWidth={1.5} strokeDasharray="4,2"/>
        <text x={163} y={521} textAnchor="middle" fontSize={11} fontWeight={700} fill="#a5b4fc">Stage 3 Candidates 맵핑</text>
        <text x={163} y={539} textAnchor="middle" fontSize={9.5} fill="#5850b0">S2 결과 → S3 소분류 목록</text>

        <line x1={485} y1={444} x2={485} y2={498} stroke="#818cf8" strokeWidth={1.8} strokeDasharray="4,2" markerEnd="url(#ah-indigo)"/>
        <rect x={392} y={500} width={186} height={50} rx={8} fill="#13103a" stroke="#818cf8" strokeWidth={1.5} strokeDasharray="4,2"/>
        <text x={485} y={521} textAnchor="middle" fontSize={11} fontWeight={700} fill="#a5b4fc">Stage 4 Candidates 맵핑</text>
        <text x={485} y={539} textAnchor="middle" fontSize={9.5} fill="#5850b0">S3 결과 → S4 증상 목록</text>

        {/* cross: S3 candidates → S3 classifier */}
        <path d="M 254 525 C 334 525 360 416 380 414" fill="none" stroke="#818cf8" strokeWidth={1.5} strokeDasharray="4,2" markerEnd="url(#ah-indigo)"/>
        <text x={278} y={514} fontSize={9} fill="#6366f1">candidates 전달</text>

        {/* cross: S4 candidates → S4 classifier */}
        <path d="M 578 525 C 652 525 694 416 712 414" fill="none" stroke="#818cf8" strokeWidth={1.5} strokeDasharray="4,2" markerEnd="url(#ah-indigo)"/>
        <text x={608} y={514} fontSize={9} fill="#6366f1">candidates 전달</text>

        {/* ── FINAL OUTPUT ── */}
        <line x1={819} y1={444} x2={819} y2={595} stroke="#f472b6" strokeWidth={2.2} markerEnd="url(#ah-pink)"/>
        <path d="M 163 550 L 163 618 C 163 640 400 658 622 658" fill="none" stroke="#3b82f6" strokeWidth={1} strokeDasharray="5,3" markerEnd="url(#ah-blue)"/>
        <path d="M 485 550 L 485 614 C 485 640 590 656 620 656" fill="none" stroke="#22c55e" strokeWidth={1} strokeDasharray="5,3" markerEnd="url(#ah-green)"/>

        <g filter="url(#glow)">
          <rect x={622} y={597} width={296} height={68} rx={12} fill="#1a0a20" stroke="#ec4899" strokeWidth={2.2}/>
          <text x={770} y={622} textAnchor="middle" fontSize={13} fontWeight={800} fill="#f9a8d4">🏥 최종 Pre-KTAS 분류 결정</text>
          <text x={770} y={642} textAnchor="middle" fontSize={10.5} fill="#9c4a7c">Stage 4 증상 기반 KTAS 등급 (1 ~ 5)</text>
          <text x={770} y={656} textAnchor="middle" fontSize={9.5} fill="#7a3060">최종 중증도 분류 출력</text>
        </g>

        {/* Feedback loop */}
        <path d="M 82 400 C 20 336 18 112 372 48" fill="none" stroke="#fbbf24" strokeWidth={1.4} strokeDasharray="5,3" markerEnd="url(#ah-yellow)"/>
        <text x={9} y={248} fontSize={9.5} fill="#fbbf24" transform="rotate(-90, 9, 248)">사용자 답변 입력</text>

        {/* Section labels */}
        <text x={15} y={380} fontSize={8.5} fontWeight={800} fill="#3b82f6" letterSpacing={1}>STAGE 2</text>
        <text x={330} y={380} fontSize={8.5} fontWeight={800} fill="#22c55e" letterSpacing={1}>STAGE 3</text>
        <text x={660} y={380} fontSize={8.5} fontWeight={800} fill="#f97316" letterSpacing={1}>STAGE 4</text>
      </svg>

      {/* Legend */}
      <div style={{ display: "flex", justifyContent: "center", flexWrap: "wrap", gap: 14, marginTop: 22 }}>
        {[
          { color: "#38bdf8", bg: "#0c1a2e", label: "Streaming Input" },
          { color: "#a78bfa", bg: "#1a0d38", label: "재평가 판단 (LLM Router)" },
          { color: "#3b82f6", bg: "#0d1f40", label: "Stage 2 노드/분류기" },
          { color: "#22c55e", bg: "#0d2818", label: "Stage 3 노드/분류기" },
          { color: "#f97316", bg: "#2a1000", label: "Stage 4 노드/분류기" },
          { color: "#fbbf24", bg: "#1c1400", label: "추가 질문 생성 (점선)", dashed: true },
          { color: "#818cf8", bg: "#13103a", label: "Candidates 맵핑 (점선)", dashed: true },
          { color: "#ec4899", bg: "#1a0a20", label: "최종 KTAS 출력" },
        ].map(({ color, bg, label, dashed }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "#94a3b8" }}>
            <div style={{
              width: 13, height: 13, borderRadius: 3,
              border: `2px ${dashed ? "dashed" : "solid"} ${color}`,
              background: bg
            }} />
            {label}
          </div>
        ))}
      </div>
    </div>
  );
}