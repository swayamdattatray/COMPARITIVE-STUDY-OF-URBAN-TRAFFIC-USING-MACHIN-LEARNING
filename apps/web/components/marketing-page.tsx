"use client";

import { motion } from "framer-motion";
import { Activity, Award, BarChart3, CalendarCheck, CheckCircle2, ChevronRight, Code2, KeyRound, Layers3, type LucideIcon, Play, Rocket, ShieldCheck, Sparkles, Trophy, UserRound } from "lucide-react";
import { activities, courseModules, dashboardStats, trainers } from "@/lib/data";

const fade = { hidden: { opacity: 0, y: 24 }, visible: { opacity: 1, y: 0 } };

export function MarketingPage() {
  return (
    <main className="min-h-screen overflow-hidden px-5 py-6 text-slate-950 dark:text-white md:px-8">
      <nav className="mx-auto flex max-w-7xl items-center justify-between rounded-3xl border border-white/10 bg-white/70 px-5 py-4 shadow-2xl shadow-slate-950/10 backdrop-blur dark:bg-slate-950/60">
        <div className="flex items-center gap-3 font-black tracking-tight"><span className="rounded-2xl bg-gradient-to-br from-violet-500 to-cyan-400 p-2"><Sparkles className="h-5 w-5 text-white" /></span>CodeMentor AI</div>
        <div className="hidden items-center gap-6 text-sm text-slate-600 dark:text-slate-300 md:flex"><a href="#trainers">AI Trainers</a><a href="#dashboard">Dashboard</a><a href="#assessment">Testing</a><a href="#admin">Admin</a></div>
        <button className="rounded-2xl bg-slate-950 px-4 py-2 text-sm font-bold text-white dark:bg-white dark:text-slate-950">Launch SaaS</button>
      </nav>

      <section className="mx-auto grid max-w-7xl gap-10 py-16 lg:grid-cols-[1.05fr_.95fr] lg:items-center">
        <motion.div initial="hidden" animate="visible" variants={fade} transition={{ duration: .7 }}>
          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-cyan-300/30 bg-cyan-300/10 px-4 py-2 text-sm font-semibold text-cyan-300"><Rocket className="h-4 w-4" /> AI-powered programming academy</div>
          <h1 className="text-5xl font-black leading-tight tracking-tight md:text-7xl">Premium EdTech SaaS for adaptive coding mastery.</h1>
          <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-600 dark:text-slate-300">CodeMentor AI combines specialized AI trainers, structured courses, milestone testing, a multilingual coding playground, gamified progress, and verified certificates in one production-ready learning platform.</p>
          <div className="mt-8 flex flex-col gap-3 sm:flex-row"><button className="rounded-2xl bg-gradient-to-r from-violet-500 to-cyan-400 px-6 py-4 font-bold text-white shadow-glow">Start learning <ChevronRight className="inline h-4 w-4" /></button><button className="rounded-2xl border border-white/15 px-6 py-4 font-bold"><Play className="mr-2 inline h-4 w-4" />View demo</button></div>
        </motion.div>
        <motion.div initial={{ opacity: 0, scale: .95 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: .7, delay: .15 }} className="glass rounded-[2rem] p-5">
          <div className="rounded-[1.5rem] bg-slate-950 p-5 text-white shadow-2xl">
            <div className="flex items-center justify-between"><div><p className="text-sm text-cyan-300">AI Trainer Session</p><h2 className="text-2xl font-black">Python Development</h2></div><span className="rounded-full bg-emerald-400/15 px-3 py-1 text-sm text-emerald-300">Live</span></div>
            <div className="mt-6 space-y-4"><Chat who="AI" text="Your weak area is recursion. I created a 3-step revision path and two coding drills." /><Chat who="You" text="Show me a practical example." /><Chat who="AI" text="Let's build a recursive directory scanner, then convert it to an iterative stack solution." /></div>
            <div className="mt-6 grid grid-cols-3 gap-3 text-center"><Mini label="XP" value="12,840" /><Mini label="Level" value="18" /><Mini label="Streak" value="21d" /></div>
          </div>
        </motion.div>
      </section>

      <Section id="trainers" eyebrow="Specialized trainers" title="Ten AI mentors for the skills students actually need.">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">{trainers.map((trainer) => <motion.article whileHover={{ y: -6 }} key={trainer.name} className="glass rounded-3xl p-5"><trainer.icon className="mb-4 h-7 w-7 text-cyan-300" /><h3 className="font-black">{trainer.name}</h3><p className="mt-2 text-sm text-slate-600 dark:text-slate-300">{trainer.focus}</p><div className="mt-4 h-2 rounded-full bg-white/10"><div className="h-2 rounded-full bg-gradient-to-r from-violet-500 to-cyan-400" style={{ width: `${trainer.progress}%` }} /></div></motion.article>)}</div>
      </Section>

      <Section id="dashboard" eyebrow="Student command center" title="A polished dashboard for learning momentum.">
        <div className="grid gap-5 lg:grid-cols-[.85fr_1.15fr]"><div className="glass rounded-3xl p-6"><UserRound className="h-8 w-8 text-violet-300" /><h3 className="mt-4 text-2xl font-black">Alex Morgan</h3><p className="text-slate-600 dark:text-slate-300">Full Stack Web Development • Student</p><div className="mt-6 grid grid-cols-2 gap-3">{dashboardStats.map((s) => <Mini key={s.label} label={s.label} value={s.value} detail={s.detail} />)}</div></div><div className="glass rounded-3xl p-6"><h3 className="mb-4 flex items-center gap-2 text-xl font-black"><Activity className="text-cyan-300" /> Recent activities & recommendations</h3>{activities.map((a) => <div key={a} className="mb-3 rounded-2xl border border-white/10 bg-white/10 p-4 text-sm"><CheckCircle2 className="mr-2 inline h-4 w-4 text-emerald-300" />{a}</div>)}</div></div>
      </Section>

      <Section id="courses" eyebrow="Course management" title="Every course ships with levels, resources, chat, playground, and progress.">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{courseModules.map((m) => <div key={m} className="rounded-2xl border border-white/10 bg-white/10 p-4 font-semibold"><Layers3 className="mr-2 inline h-4 w-4 text-cyan-300" />{m}</div>)}</div>
      </Section>

      <Section id="assessment" eyebrow="Automated assessment engine" title="Milestone and weekly Saturday tests powered by AI analysis.">
        <div className="grid gap-5 md:grid-cols-3"><Feature icon={CalendarCheck} title="Milestone tests" text="Unlock automatically at 25%, 50%, 75%, and 100% with MCQs, coding challenges, and practical assignments." /><Feature icon={BarChart3} title="Performance analysis" text="Detects weak areas, suggests revision plans, generates custom quizzes, and predicts completion timelines." /><Feature icon={Code2} title="Coding playground" text="Run Python, C, PHP, and JavaScript with AI code review, error detection, and bug-fix suggestions." /></div>
      </Section>

      <Section id="admin" eyebrow="SaaS operations" title="Admin, security, certificates, and gamification are built in.">
        <div className="grid gap-5 md:grid-cols-4"><Feature icon={Trophy} title="Gamification" text="XP, levels, badges, streaks, leaderboards, and achievements." /><Feature icon={Award} title="Certificates" text="Auto-generated after 100% completion and final assessment with unique IDs and QR verification." /><Feature icon={KeyRound} title="Auth & RBAC" text="JWT, Google login, email verification, and Student/Trainer/Admin roles." /><Feature icon={ShieldCheck} title="Admin panel" text="Manage students, courses, AI trainers, tests, analytics, announcements, and reports." /></div>
      </Section>
    </main>
  );
}

function Section({ id, eyebrow, title, children }: { id: string; eyebrow: string; title: string; children: React.ReactNode }) { return <section id={id} className="mx-auto max-w-7xl py-12"><p className="mb-3 font-bold uppercase tracking-[.3em] text-cyan-300">{eyebrow}</p><h2 className="mb-8 max-w-4xl text-3xl font-black md:text-5xl">{title}</h2>{children}</section>; }
function Chat({ who, text }: { who: string; text: string }) { return <div className="rounded-2xl border border-white/10 bg-white/10 p-4"><span className="text-xs font-bold text-cyan-300">{who}</span><p className="mt-1 text-sm text-slate-200">{text}</p></div>; }
function Mini({ label, value, detail }: { label: string; value: string; detail?: string }) { return <div className="rounded-2xl border border-white/10 bg-white/10 p-4"><p className="text-xs text-slate-400">{label}</p><p className="text-xl font-black">{value}</p>{detail && <p className="text-xs text-cyan-300">{detail}</p>}</div>; }
function Feature({ icon: Icon, title, text }: { icon: LucideIcon; title: string; text: string }) { return <article className="glass rounded-3xl p-6"><Icon className="mb-4 h-8 w-8 text-violet-300" /><h3 className="text-xl font-black">{title}</h3><p className="mt-3 text-sm leading-6 text-slate-600 dark:text-slate-300">{text}</p></article>; }
