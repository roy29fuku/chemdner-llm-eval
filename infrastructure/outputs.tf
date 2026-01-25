output "instance_name" {
  value       = google_compute_instance.biobert_training.name
  description = "Name of the training instance"
}

output "instance_zone" {
  value       = google_compute_instance.biobert_training.zone
  description = "Zone of the training instance"
}

output "project_id" {
  value       = var.project_id
  description = "GCP Project ID"
}

output "scp_upload_command" {
  value       = "gcloud compute scp --recurse LOCAL_PATH ${var.instance_name}:~/chemdner/ --zone=${var.zone} --project=${var.project_id}"
  description = "SCP command to upload files"
}

output "scp_download_command" {
  value       = "gcloud compute scp --recurse ${var.instance_name}:~/chemdner/outputs/ LOCAL_PATH --zone=${var.zone} --project=${var.project_id}"
  description = "SCP command to download results"
}
