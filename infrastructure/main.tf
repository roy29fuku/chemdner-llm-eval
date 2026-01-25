terraform {
  required_version = ">= 1.0.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Compute Instance for BioBERT training
resource "google_compute_instance" "biobert_training" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      # Deep Learning VM with PyTorch 2.x and CUDA
      image = "projects/deeplearning-platform-release/global/images/family/pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
      size  = var.disk_size
      type  = "pd-ssd"
    }
  }

  network_interface {
    network = "default"
    access_config {
      # Ephemeral public IP
    }
  }

  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false
  }

  metadata = {
    install-nvidia-driver = "True"
  }

  metadata_startup_script = file("${path.module}/startup-script.sh")

  tags = ["biobert-training", "allow-ssh"]

  labels = {
    purpose     = "biobert-training"
    project     = "chemdner"
    environment = "training"
  }

  service_account {
    scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

# Firewall rule to allow SSH
resource "google_compute_firewall" "allow_ssh" {
  name    = "allow-ssh-biobert"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["allow-ssh"]
}

# Output the instance external IP
output "instance_ip" {
  value       = google_compute_instance.biobert_training.network_interface[0].access_config[0].nat_ip
  description = "External IP of the training instance"
}

output "ssh_command" {
  value       = "gcloud compute ssh ${var.instance_name} --zone=${var.zone} --project=${var.project_id}"
  description = "SSH command to connect to the instance"
}
