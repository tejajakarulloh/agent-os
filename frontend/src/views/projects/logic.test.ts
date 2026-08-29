import { describe, expect, it } from 'vitest'
import {
  filterProjects,
  filterSessionsByProject,
  groupProjectSessionsByAgent,
  knowledgeExcerpt,
  projectId,
  projectName,
  projectSessionCount,
  sessionProjectId,
  sessionsInProject,
  sortProjects,
} from './logic'

describe('project row accessors', () => {
  it('reads ids and names from both casings', () => {
    expect(projectId({ project_id: 'a1' })).toBe('a1')
    expect(projectId({ projectId: 'b2' })).toBe('b2')
    expect(projectId({})).toBe('')
    expect(projectName({ name: 'Research' })).toBe('Research')
    expect(projectName({ project_id: 'abcdef1234567890' })).toBe('abcdef12')
    expect(projectSessionCount({ session_count: 3 })).toBe(3)
    expect(projectSessionCount({ sessionCount: 2 })).toBe(2)
    expect(projectSessionCount({})).toBe(0)
  })

  it('reads a session row project id from both casings', () => {
    expect(sessionProjectId({ project_id: 'p1' })).toBe('p1')
    expect(sessionProjectId({ projectId: 'p2' })).toBe('p2')
    expect(sessionProjectId({ project_id: null })).toBe('')
    expect(sessionProjectId({})).toBe('')
  })
})

describe('sortProjects', () => {
  it('orders by updated_at desc with name tiebreak, without mutating', () => {
    const input = [
      { project_id: 'a', name: 'Beta', updated_at: 100 },
      { project_id: 'b', name: 'Alpha', updated_at: 100 },
      { project_id: 'c', name: 'Newest', updated_at: 300 },
    ]
    const sorted = sortProjects(input)
    expect(sorted.map((p) => p.name)).toEqual(['Newest', 'Alpha', 'Beta'])
    expect(input[0]!.name).toBe('Beta')
  })
})

describe('filterProjects', () => {
  const projects = [
    { project_id: 'a', name: 'Token research', knowledge: 'solana pools' },
    { project_id: 'b', name: 'Docs', knowledge: '' },
  ]
  it('matches name and knowledge case-insensitively', () => {
    expect(filterProjects(projects, 'TOKEN').map((p) => p.project_id)).toEqual(['a'])
    expect(filterProjects(projects, 'solana').map((p) => p.project_id)).toEqual(['a'])
    expect(filterProjects(projects, '')).toHaveLength(2)
  })
})

describe('sessionsInProject / filterSessionsByProject', () => {
  const sessions = [
    { key: 'k1', project_id: 'p1', updatedAt: 100 },
    { key: 'k2', project_id: 'p1', updatedAt: 300 },
    { key: 'k3', projectId: 'p2', updated_at: 200 },
    { key: 'k4', updatedAt: 400 },
  ]

  it('lists a project sessions newest first across both timestamp casings', () => {
    expect(sessionsInProject(sessions, 'p1').map((s) => s.key)).toEqual(['k2', 'k1'])
  })

  it("filters by 'all', 'none', and a project id", () => {
    expect(filterSessionsByProject(sessions, 'all')).toHaveLength(4)
    expect(filterSessionsByProject(sessions, 'none').map((s) => s.key)).toEqual(['k4'])
    expect(filterSessionsByProject(sessions, 'p2').map((s) => s.key)).toEqual(['k3'])
  })
})

describe('groupProjectSessionsByAgent', () => {
  it('buckets by agent alphabetically, each bucket newest first across both timestamp casings', () => {
    const sessions = [
      { key: 'k1', agent_id: 'zeta', updatedAt: 100 },
      { key: 'k2', agentId: 'alpha', updatedAt: 100 },
      { key: 'k3', agent_id: 'alpha', updatedAt: 300 },
      { key: 'k4', updated_at: 50 },
    ]
    const groups = groupProjectSessionsByAgent(sessions)
    expect(groups.map((g) => g.agentId)).toEqual(['alpha', 'main', 'zeta'])
    expect(groups[0]!.items.map((s) => s.key)).toEqual(['k3', 'k2'])
    expect(groups[1]!.items.map((s) => s.key)).toEqual(['k4'])
  })
})

describe('knowledgeExcerpt', () => {
  it('flattens whitespace and truncates with an ellipsis', () => {
    expect(knowledgeExcerpt('a  b\nc')).toBe('a b c')
    const long = 'x'.repeat(200)
    const excerpt = knowledgeExcerpt(long, 20)
    expect(excerpt).toHaveLength(20)
    expect(excerpt.endsWith('…')).toBe(true)
  })
})
