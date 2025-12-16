Set-StrictMode -Version Latest

$script:repoRoot = $null
$script:agentsPath = $null
$script:agents = @()

Describe "Adapted Copilot agents attribution" {
    BeforeAll {
        $script:repoRoot = Split-Path -Path (Split-Path -Path (Split-Path -Path $PSScriptRoot -Parent) -Parent) -Parent
        $script:agentsPath = Join-Path -Path $repoRoot -ChildPath ".github/agents"
        $script:agents = Get-ChildItem -Path $script:agentsPath -Filter "*-adjusted.agent.md" -File
    }

    It "finds adapted agent files" {
        $script:agents | Should -Not -BeNullOrEmpty
    }

    foreach ($agent in $script:agents) {
        Context -Name $agent.Name {
            BeforeAll {
                $script:content = Get-Content -Path $agent.FullName -Raw
            }

            It "includes repo policy precedence section" {
                $script:content | Should -Match "Repo Policy Compliance \(Highest Priority\)"
            }

            It "includes provenance header referencing awesome-copilot" {
                $script:content | Should -Match "Adapted from github/awesome-copilot"
            }

            It "links to THIRD_PARTY_NOTICES" {
                $script:content | Should -Match "THIRD_PARTY_NOTICES.md"
            }
        }
    }
}
